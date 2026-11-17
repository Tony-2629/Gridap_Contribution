# -------------------------------------------------
# 1. Packages
# -------------------------------------------------
using Gridap
using Gridap.Fields
using Gridap.Geometry
using Gridap.CellData
using LineSearches: BackTracking
using Statistics

# -------------------------------------------------
# 2. Geometry – three rectangles (union = full domain)
# -------------------------------------------------
L, H = 1.24, 0.6                     # total size
h_bot = 0.2*H                       # yellow bottom thickness
h_mid = 0.6*H                       # brown middle thickness
h_top = 0.2*H                       # yellow top thickness
domain = (0, L, 0, H)

#3---6---4
#|       |
#7   9   8
#|       |
#1---5---2

# -------------------------------------------------
# 3. Discretisation
# -------------------------------------------------
partition = (31, 15)
model = CartesianDiscreteModel(domain, partition)

# -------------------------------------------------
# 4. Tag cells by centroid Y-coordinate
# -------------------------------------------------
Ω = Triangulation(model)
cell_points = get_cell_coordinates(Ω)          # 4 corner points per cell
cell_centroids = lazy_map(mean, cell_points)   # ← one centroid per cell

# Extract Y-coordinate of each centroid
centroid_y = get_array(lazy_map(p -> p[2], cell_centroids))


# Material IDs (1 = yellow, 2 = brown)
material_id = zeros(Int, num_cells(model))
for (i, q) in enumerate(cell_centroids)
    (x,y) = q 
    if ((y ≤ h_bot || y ≥ (h_bot + h_mid) || x ≤ 0.2*H) && x ≤ 1.2)    # yellow layers
        material_id[i] = 1
    else                                                 # brown middle
        material_id[i] = 2
    end
end
material_field = CellField(material_id, Ω)   # for VTK

# -------------------------------------------------
# 5. Material parameters (per cell)
# -------------------------------------------------
# Yellow (stiff)   → λ₁, μ₁
λ₁ = 100.0;  μ₁ = 1.0
# Brown  (soft)    → λ₂, μ₂
λ₂ = λ₁*1e-1;  μ₂ = μ₁*1e-1
kᵣ = 1e-7*L*(λ₁+(2/3)*μ₁)
# -------------------------------------------------
# 6. Constitutive law – Neo-Hookean (per cell)
# -------------------------------------------------
F(∇u) = one(∇u) + ∇u'
C(∇u) = F(∇u)' ⋅ F(∇u)
J(∇u) = sqrt(det(C(∇u)))

# Virtual Green-Lagrange strain increment
dE(∇v, ∇u) = 0.5 * (∇v ⋅ F(∇u) + F(∇u)' ⋅ ∇v')


# Tangent modulus dS/d∇du (material part only)
function dS_bimat(∇du, ∇u, tag)
    if tag == 1
        Cinv = inv(C(∇u))
        _dE  = dE(∇du, ∇u)
        return λ₁*(Cinv ⊙ _dE)*Cinv +
               2*(μ₁ - λ₁*log(J(∇u)))*Cinv ⋅ _dE ⋅ (Cinv')
    else
        Cinv = inv(C(∇u))
        _dE  = dE(∇du, ∇u)
        return λ₂*(Cinv ⊙ _dE)*Cinv +
               2*(μ₂ - λ₂*log(J(∇u)))*Cinv ⋅ _dE ⋅ (Cinv')
    end
end

# 2nd Piola-Kirchhoff stress (depends on λ, μ of the cell)
function S_bimat(∇u, tag)
    if tag == 1                      # yellow
        Cinv = inv(C(∇u))
        return μ₁*(one(∇u) - Cinv) + λ₁*log(J(∇u))*Cinv
    else                             # brown
        Cinv = inv(C(∇u))
        return μ₂*(one(∇u) - Cinv) + λ₂*log(J(∇u))*Cinv
    end
end




# Cauchy stress (post-processing)
function σ_bimat(∇u, tag)
    if tag == 1
        return (1.0/J(∇u)) * F(∇u) ⋅ S_bimat(∇u,1) ⋅ F(∇u)'
    else
        return (1.0/J(∇u)) * F(∇u) ⋅ S_bimat(∇u,2) ⋅ F(∇u)'
    end
end

# Hessian operation
function scale(∇u)    
    return exp(-5.0*J(∇u))
end


# -------------------------------------------------
# FIND NODE BY COORDINATE (tolerance = 1e-6)
# -------------------------------------------------
tol = 1e-6
tip_point = Point(1.2, 0.6)  # (x,y) of the node

# Get all node coordinates
node_coords = Gridap.Geometry.get_node_coordinates(model) 


# Find node index
let tip_node_id = nothing
for (i, p) in enumerate(node_coords)
    if norm(p - tip_point) < tol
        tip_node_id = i
        break
    end
end

@assert tip_node_id !== nothing "Tip node not found – check coordinates / tolerance"
    # ---- expose only the id to the outer scope ----
    global tip_node_id_global = tip_node_id
end

println("Prescribed displacement at node $tip_node_id_global: (1.2, 0.5)")

# -------------------------------------------------
# 7. Boundary tags
# -------------------------------------------------
labels = get_face_labeling(model)
entity = num_entities(labels)+1
labels.d_to_dface_to_entity[1][tip_node_id_global] = entity
add_tag!(labels,"Tip",[entity])
add_tag_from_tags!(labels, "clamped", [1, 3, 7])   # left edge + corners

# -------------------------------------------------
# 8. Integration measures
# -------------------------------------------------
order = 2
degree = 2*order
dΩ = Measure(Ω, degree+2)

# -------------------------------------------------
# 9. FE spaces
# -------------------------------------------------
reffe = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
V = TestFESpace(model, reffe, conformity=:H1,
                dirichlet_tags=["clamped", "Tip"])
# Trial space will be built inside the incremental loop

reg_mask = CellField(material_id .== 2, Ω)

# -------------------------------------------------
# 10. Weak form – material parameters vary per cell
# -------------------------------------------------
res(u,v) = ∫(
    (dE ∘ (∇(v), ∇(u))) ⊙ (S_bimat ∘ (∇(u), material_field))
    + (∇∇(u) ⊙ ∇∇(v))*kᵣ*reg_mask 
)dΩ

jac_mat(u,du,v) = ∫(
    (dE ∘ (∇(v), ∇(u))) ⊙ (dS_bimat ∘ (∇(du), ∇(u), material_field))
)dΩ

jac_geo(u,du,v) = ∫(
    ∇(v) ⊙ ( (S_bimat ∘ (∇(u), material_field)) ⋅ ∇(du) )
)dΩ

jac_reg(du,v) = ∫( (∇∇(du) ⊙ ∇∇(v)) * kᵣ * reg_mask )dΩ

jac(u, du, v) = jac_mat(u, du, v) + jac_geo(u, du, v) + jac_reg(du,v)

# -------------------------------------------------
# 11. Solver
# -------------------------------------------------
nls = NLSolver(show_trace=true, method=:newton,
               linesearch=BackTracking())
solver = FESolver(nls)

# -------------------------------------------------
# 12. Incremental prescribed displacement
# -------------------------------------------------
function run_step(x0, disp_y, step, nsteps, cache)
    # Dirichlet values
    g_clamped = VectorValue(0.0, 0.0)
    g_loaded  = VectorValue(0.0, disp_y)          # vertical displacement

    U = TrialFESpace(V, [g_clamped, g_loaded])

    op = FEOperator(res, jac, U, V)

    println("\n=== Step $step / $nsteps – disp_y = $disp_y ===\n")
    uh = FEFunction(U, x0)
    uh, cache = solve!(uh, solver, op, cache)

    # ----- VTK output (includes material_id) -----
    mkpath("TMC_C_shape")
    writevtk(
        Ω,
        "TMC_C_shape/results_$(lpad(step,3,'0'))";
        cellfields = [
            "uh"         => uh,
            "sigma"  => σ_bimat ∘ (ε(uh),material_field),
            "material_id"=> material_field
        ]
    )
    return get_free_dof_values(uh), cache
end

function run_all()
    disp_max = -0.7               # final vertical displacement (down)
    nsteps   = 100
    x0 = zeros(Float64, num_free_dofs(V))
    cache = nothing
    for step in 1:nsteps
        disp_y = step * disp_max / nsteps
        x0, cache = run_step(x0, disp_y, step, nsteps, cache)
    end
end

# -------------------------------------------------
# 13. Execute
# -------------------------------------------------
run_all()
println("\nFinished – open folder 'TMC_C_shape' in Paraview")