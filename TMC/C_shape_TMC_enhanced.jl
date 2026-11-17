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
L, H = 1.0, 0.5                     # total size
h_bot = 0.2*H                       # yellow bottom thickness
h_mid = 0.6*H                       # brown middle thickness
h_top = 0.2*H                       # yellow top thickness

domain = (0, L, 0, H)

# -------------------------------------------------
# 3. Discretisation
# -------------------------------------------------
nx, ny = 80, 20                     # adjust for finer mesh
model = CartesianDiscreteModel(domain, (nx, ny))

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
    if y ≤ h_bot || y ≥ (h_bot + h_mid) || x ≤ 0.2*H     # yellow layers
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
λ₂ = λ₁*10^(-5);  μ₂ = μ₁*10^(-5)
η = 1e-5
# -------------------------------------------------
# 6. Constitutive law – Neo-Hookean (per cell)
# -------------------------------------------------
F(∇u) = one(∇u) + ∇u'
C(∇u) = F(∇u)' ⋅ F(∇u)
J(∇u) = sqrt(det(C(∇u)))

# Virtual Green-Lagrange strain increment
dE(∇v, ∇u) = 0.5 * (∇v ⋅ F(∇u) + F(∇u)' ⋅ ∇v')

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

# -------------------------------------------------
# 7. Boundary tags
# -------------------------------------------------
labels = get_face_labeling(model)
add_tag_from_tags!(labels, "clamped", [1, 3, 7])   # left edge + corners
add_tag_from_tags!(labels, "loaded",  [4])   # right edge + corners

# -------------------------------------------------
# 8. Integration measures
# -------------------------------------------------
degree = 4
dΩ = Measure(Ω, degree+2)

# -------------------------------------------------
# 9. FE spaces
# -------------------------------------------------
reffe = ReferenceFE(lagrangian, VectorValue{2,Float64}, 2)
V = TestFESpace(model, reffe, conformity=:H1,
                dirichlet_tags=["clamped", "loaded"])
# Trial space will be built inside the incremental loop

# -------------------------------------------------
# 10. Weak form – material parameters vary per cell
# -------------------------------------------------
res(u,v) = ∫(
    (dE ∘ (∇(v), ∇(u))) ⊙ (S_bimat ∘ (∇(u), material_field))
)dΩ

jac_mat(u,du,v) = ∫(
    (dE ∘ (∇(v), ∇(u))) ⊙ (dS_bimat ∘ (∇(du), ∇(u), material_field))
)dΩ

jac_geo(u,du,v) = ∫(
    ∇(v) ⊙ ( (S_bimat ∘ (∇(u), material_field)) ⋅ ∇(du) )
)dΩ

jac(u, du, v) = jac_mat(u, du, v) + jac_geo(u, du, v)

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
    mkpath("output_two_mat")
    writevtk(
        Ω,
        "output_two_mat/results_$(lpad(step,3,'0'))";
        cellfields = [
            "uh"         => uh,
            "material_id"=> material_field
        ]
    )
    return get_free_dof_values(uh), cache
end

function run_all()
    disp_max = -0.10               # final vertical displacement (down)
    nsteps   = 20
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
println("\nFinished – open folder 'output_two_mat' in Paraview")