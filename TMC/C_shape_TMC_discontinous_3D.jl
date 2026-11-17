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
L, H, T = 0.8, 0.28, 0.2                     # Length, Height, thickness (extrusion in Z)
h_bot = 0.2*T                                # yellow bottom thickness
h_mid = 0.6*H                                # brown middle thickness
h_top = 0.2*H                                # yellow top thickness
domain = (0, L, 0, H, 0, T)



# -------------------------------------------------
# 3. Discretisation
# -------------------------------------------------
partition = (20, 7, 5)                         # (nx, ny, nz)
model = CartesianDiscreteModel(domain, partition)

d = L/20;

# -------------------------------------------------
# 4. Tag cells by centroid Y-coordinate
# -------------------------------------------------
Ω = Triangulation(model)
cell_points = get_cell_coordinates(Ω)          # 4 corner points per cell
cell_centroids = lazy_map(mean, cell_points)   # ← one centroid per cell


# Material IDs (1 = yellow, 2 = brown)
material_id = zeros(Int, num_cells(model))
for (i, c) in enumerate(cell_centroids)
    (x, y, z) = c 
    if ((y ≤ h_bot || y ≥ (0.24) || x ≤ 0.2*T) && x ≤ 1.2)    # yellow layers
        material_id[i] = 1
    else                                                 # brown middle
        material_id[i] = 2
    end
end
material_field = CellField(material_id, Ω)   # for VTK
# -------------------------------------------------
#  Write VTK file — open immediately in Paraview!
# -------------------------------------------------
#writevtk(Ω, "C_shape_3D_geometry",
#    cellfields = ["material_id" => material_field])

# -------------------------------------------------
# 5. Material parameters (per cell)
# -------------------------------------------------
# Regularisation parameters
γ  = 1.0e-7              # weight
β₁ = 1000000.0             # (α - p)^2 coefficient
αᵣ = 10000.0               # ∇p⋅∇p length scale

# Yellow (stiff)   → K₁, μ₁
K₁ = 5.0/3.0;  μ₁ = 5/14;
λ₁ = K₁ - (2/3)*μ₁;

# Brown  (soft)    → K₂, μ₂
K₂ = K₁*γ;  μ₂ = μ₁*γ; λ₂ = λ₁*γ; 



# -------------------------------------------------
# 6. Constitutive law – Neo-Hookean (per cell)
# -------------------------------------------------
F(∇u) = one(∇u) + ∇u'
C(∇u) = F(∇u)' ⋅ F(∇u)
J(∇u) = sqrt(det(C(∇u)))

# Virtual Green-Lagrange strain increment
dE(∇v, ∇u) = 0.5 * (∇v ⋅ F(∇u) + F(∇u)' ⋅ ∇v')


# Tangent modulus dS/d∇du (material part only)
function dS(∇du, ∇u, tag)
    K, μ = (tag == 1) ? (K₁, μ₁) : (K₂, μ₂)
    Cinv = inv(C(∇u))
    _dE  = dE(∇du, ∇u)
    return K*(Cinv ⊙ _dE)*Cinv +
            3*(μ - K*log(J(∇u)))*Cinv ⋅ _dE ⋅ (Cinv')

end

# 1st Piola-Kirchhoff stress (depends on K, μ of the cell)
#function P(∇u, tag)
#    K, μ = (tag == 1) ? (K₁, μ₁) : (K₂, μ₂)
#    Fval = F(∇u)
#    Jval = J(∇u)
#    lnJ = log(Jval)
#    Cinv = inv(C(∇u))
#    Finv = inv(F(∇u))
#    I1 =tr(C(∇u))
#    P = K*log(J(∇u))*Finv' + μ*J(∇u)^(-2/3)*(Fval - (1/3)*I1*Finv')
#    return P
#end

# 2nd Piola-Kirchhoff stress (depends on K, μ of the cell)
#function S(∇u, tag)
#    K, μ = (tag == 1) ? (K₁, μ₁) : (K₂, μ₂)
#    Cinv = inv(C(∇u))
#    I1 =tr(C(∇u))
#    return K*log(J(∇u))*Cinv + μ*J(∇u)^(-2/3)*(one(∇u)-(1/3)*I1*Cinv)
#end



function S(∇u, tag)
    λ, μ = (tag == 1) ? (λ₁, μ₁) : (λ₂, μ₂)
    Cinv = inv(C(∇u))
    return μ*(one(∇u) - Cinv) + λ*log(J(∇u))*Cinv
end

#function P(∇u, tag)
#    return F(∇u) ⋅ S(∇u, tag)
#end

#function S(∇u, tag)
    # Compute deformation gradient F and right Cauchy-Green tensor C
#    F = one(∇u) + ∇u'           # F = I + ∇uᵀ
#    C = F' ⋅ F                   # C = FᵀF
#    J = det(F)                   # J = det(F) = sqrt(det(C))
#    J = max(J, 1e-12)
#    Cinv = inv(C)                # Inverse of C
#    trC = tr(C)                  # Trace of C

    # Select material parameters based on tag
#    K = (tag == 1) ? K₁ : K₂     # Bulk modulus
#    μ = (tag == 1) ? μ₁ : μ₂     # Shear modulus

    # Volumetric part: K (ln J) C⁻¹
#    S_vol = K * log(J) * Cinv

    # Isochoric (deviatoric) part: μ J⁻²/³ [I - (trC/3) C⁻¹]
#    dev_term = one(∇u) - (trC / 3.0) * Cinv
#    S_iso = μ * J^(-2.0/3.0) * dev_term

    # Total Second Piola-Kirchhoff stress
#    return S_iso
#end


#function P(∇u)
#  Fm = F(∇u)
#  Jm = J(∇u)
#  lnJ = log(Jm)
#  Finv = inv(Fm)
#  P = μ*(Fm - Finv') + λ*lnJ*Finv'
#  return P
#end


# Cauchy stress (post-processing)
function σ(∇u, tag)
        return (1.0/J(∇u)) * F(∇u) ⋅ S(∇u,tag) ⋅ F(∇u)'
end

# Hessian operation
function scale(∇u)    
    return exp(-5.0*J(∇u))
end


# -------------------------------------------------
# 7. Boundary tags
# -------------------------------------------------
# Tag the left face (x = 0); right face (x = L)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"clamped", [1,3,5,7,13,15,17,19,25]); # left face: corners (1,3,5,7); edges (13,15,17,19); others (25) 
add_tag_from_tags!(labels, "load", [4,8,20]); # Right face: corners (2,4,6,8); edges (14,16,18,20); others (26) 

# -------------------------------------------------
# 8. Integration measures
# -------------------------------------------------
order = 1
degree = 2*order
dΩ = Measure(Ω, degree+2)

# -------------------------------------------------
# 9. FE spaces
# -------------------------------------------------
reffe_u = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
V = TestFESpace(model, reffe_u, conformity=:H1,
                dirichlet_tags=["clamped", "load"])

reffe_p1 = ReferenceFE(lagrangian, Float64, order)
V_p1 = TestFESpace(model, reffe_p1, conformity=:L2)

reffe_p2 = ReferenceFE(lagrangian, Float64, order)
V_p2 = TestFESpace(model, reffe_p2, conformity=:L2)

reffe_p3 = ReferenceFE(lagrangian, Float64, order)
V_p3 = TestFESpace(model, reffe_p3, conformity=:L2)

Y = MultiFieldFESpace([V, V_p1, V_p2, V_p3])   # (v, ξ₁, ξ₂, ξ₃)

# Trial space will be built inside the incremental loop

reg_mask = CellField(material_id .== 2, Ω)

# α(∇u) = (F12 - F21)/(F11 + F22)
function α(∇u)
    F_val = F(∇u)
    a = F_val[1,2] - F_val[2,1]
    d = F_val[1,1] + F_val[2,2] + 1e-12
    return a / d
end

# G(∇u) = ∂α/∂(∇u)   ---> check this function
function G(∇u)
    F_val = F(∇u)
    n = F_val[1,2] - F_val[2,1]
    b = (F_val[1,1] + F_val[2,2])
    d = (F_val[1,1] + F_val[2,2] + 1e-12)^2    
    return TensorValue(-n/d, b/d, -b/d, -n/d )
end


# Mask: only brown (tag == 2)
reg_mask = CellField(material_id .== 2, Ω)


# ==============================================================================
# In 3D, the regularization term split into the three components which
# define the skew symmetric part of deformation gradient
# ==============================================================================
function fskew₁(∇u)
    F_val = F(∇u)
    fs₁ =  F_val[1,2] - F_val[2,1]
    return fs₁
end 

function fskew₂(∇u)
    F_val = F(∇u)
    fs₂ =  F_val[1,3] - F_val[3,1]
    return fs₂
end 

function fskew₃(∇u)
    F_val = F(∇u)
    fs₃ =  F_val[2,3] - F_val[3,2]
    return fs₃
end 

const ∂fskew₁_∂F = TensorValue(0, 1, 0, -1, 0, 0, 0, 0, 0)

const ∂fskew₂_∂F = TensorValue(0, 0, 1, 0, 0, 0, -1, 0, 0)

const ∂fskew₃_∂F = TensorValue(0, 0, 0, 0, 0, 1, 0, -1, 0)

function Reg_fskew₁(∇u, p1, ∇p1)
    fskew = fskew₁(∇u)
    appro_val = β₁*(fskew - (1/d)*p1)^2 + αᵣ*(∇p1 ⊙ ∇p1)
    return appro_val
end

function Reg_fskew₂(∇u, p2, ∇p2)
    fskew = fskew₂(∇u)
    appro_val = β₁*(fskew - (1/d)*p2)^2 + αᵣ*(∇p2 ⊙ ∇p2)
    return appro_val
end


function Reg_fskew₃(∇u, p3, ∇p3)
    fskew = fskew₃(∇u)
    appro_val = β₁*(fskew - (1/d)*p3)^2 + αᵣ*(∇p3 ⊙ ∇p3)
    return appro_val
end

# -------------------------------------------------
# 10. Weak form – material parameters vary per cell
# -------------------------------------------------

function res((u,p1,p2,p3),(v,ξ1,ξ2,ξ3))
    hyperelas = ∫((dE ∘ (∇(v), ∇(u))) ⊙ (S ∘ (∇(u), material_field)))dΩ
    #reg_u = ∫( Reg ∘ (∇(u),p)⊙∇(v)*reg_mask)dΩ
    #reg_u = ∫(( 5e-4     *(∇(u) ⊙ ∇(v) )) *reg_mask)dΩ
    #hyperelas = ∫((∇(v))⊙(P ∘ (∇(u), material_field)))dΩ 
    reg_u1 = β₁*γ*∫((fskew₁ ∘ ∇(u) - (1/d)*p1)*(∂fskew₁_∂F ⊙ ∇(v)))dΩ 
    reg_u2 = β₁*γ*∫((fskew₂ ∘ ∇(u) - (1/d)*p2)*(∂fskew₂_∂F ⊙ ∇(v)))dΩ
    reg_u3 = β₁*γ*∫((fskew₃ ∘ ∇(u) - (1/d)*p3)*(∂fskew₃_∂F ⊙ ∇(v)))dΩ  
    reg_p1 = ∫( γ*(β₁*( (fskew₁ ∘ ∇(u)) - (1/d)*p1 )*(-1/d)*ξ1 + αᵣ*∇(p1)⋅∇(ξ1)))dΩ
    reg_p2 = ∫( γ*(β₁*( (fskew₂ ∘ ∇(u)) - (1/d)*p2 )*(-1/d)*ξ2 + αᵣ*∇(p2)⋅∇(ξ2)))dΩ
    reg_p3 = ∫( γ*(β₁*( (fskew₃ ∘ ∇(u)) - (1/d)*p3 )*(-1/d)*ξ3 + αᵣ*∇(p3)⋅∇(ξ3)))dΩ
    return hyperelas + reg_p1 + reg_p2 + reg_p3 + reg_u1 + reg_u2 + reg_u3
end


jac_mat(u,du,v) = ∫(
    (dE ∘ (∇(v), ∇(u))) ⊙ (dS ∘ (∇(du), ∇(u), material_field))
)dΩ

jac_geo(u,du,v) = ∫(
    ∇(v) ⊙ ( (S ∘ (∇(u), material_field)) ⋅ ∇(du) )
)dΩ

jac_reg(du,v) = ∫( (∇∇(du) ⊙ ∇∇(v)) * kᵣ * reg_mask )dΩ

#jac(u, du, v) = jac_mat(u, du, v) + jac_geo(u, du, v) + jac_reg(du,v)

function jac((u, p1, p2, p3), (du, dp1, dp2, dp3), (v, ξ1, ξ2, ξ3))
    # 1. Tangent Stiffness matrix of Neo-Hookean: material + geometry
    S_curr = S ∘ (∇(u), material_field)
    dS_curr = (dS ∘ (∇(du), ∇(u), material_field))
    K_mat = ∫( (dE ∘ (∇(v), ∇(u))) ⊙ dS_curr)dΩ
    K_geom = ∫(
    (∇(v) ⋅ ∇(du)') ⊙ S_curr
    )dΩ
    K_hyper = K_mat + K_geom
    # 2. Displacement - Displacement Block (K_uu_reg)
    K_uu_1 = β₁ * γ * ∫( (∂fskew₁_∂F ⊙ ∇(du)) * (∂fskew₁_∂F ⊙ ∇(v)) )dΩ
    K_uu_2 = β₁ * γ * ∫( (∂fskew₂_∂F ⊙ ∇(du)) * (∂fskew₂_∂F ⊙ ∇(v)) )dΩ
    K_uu_3 = β₁ * γ * ∫( (∂fskew₃_∂F ⊙ ∇(du)) * (∂fskew₃_∂F ⊙ ∇(v)) )dΩ
    K_uu = K_uu_1 + K_uu_2 + K_uu_3
    # 3. Displacement - Pressure Coupling (K_up and K_pu)
    # K_up terms (derivatives of reg_u w.r.t p1, p2, p3)
    K_up1 = - (β₁ * γ / d) * ∫( dp1 * (∂fskew₁_∂F ⊙ ∇(v)) )dΩ
    K_up2 = - (β₁ * γ / d) * ∫( dp2 * (∂fskew₂_∂F ⊙ ∇(v)) )dΩ
    K_up3 = - (β₁ * γ / d) * ∫( dp3 * (∂fskew₃_∂F ⊙ ∇(v)) )dΩ
    K_up = K_up1 + K_up2 + K_up3
    # K_pu terms (derivatives of reg_p w.r.t u1, u2, u3)
    K_pu1 = - (β₁ * γ / d) * ∫( ξ1 * (∂fskew₁_∂F ⊙ ∇(du)) )dΩ
    K_pu2 = - (β₁ * γ / d) * ∫( ξ2 * (∂fskew₂_∂F ⊙ ∇(du)) )dΩ
    K_pu3 = - (β₁ * γ / d) * ∫( ξ3 * (∂fskew₃_∂F ⊙ ∇(du)) )dΩ
    K_pu = K_pu1 + K_pu2 + K_pu3
    # 4. Pressure - Pressure Diagonal Blocks (K_pp)
    K_pp1 = ∫( γ * ( (β₁ / d^2) * dp1 * ξ1 + αᵣ * ∇(dp1) ⋅ ∇(ξ1) ) )dΩ
    K_pp2 = ∫( γ * ( (β₁ / d^2) * dp2 * ξ2 + αᵣ * ∇(dp2) ⋅ ∇(ξ2) ) )dΩ
    K_pp3 = ∫( γ * ( (β₁ / d^2) * dp3 * ξ3 + αᵣ * ∇(dp3) ⋅ ∇(ξ3) ) )dΩ
    K_pp = K_pp1 + K_pp2 + K_pp3 

    # ---------------------------------------------------------------------
    # Assemble the tangent stiffness matrix
    # ---------------------------------------------------------------------

    return K_hyper + K_uu + K_up + K_pu + K_pp
end


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
    g_clamped = VectorValue(0.0, 0.0, 0.0)
    g_loaded  = VectorValue(0.0, disp_y, 0.0)          # vertical displacement

    U = TrialFESpace(V, [g_clamped, g_loaded])
    U_p1 = TrialFESpace(V_p1)
    U_p2 = TrialFESpace(V_p2)
    U_p3 = TrialFESpace(V_p3)


    # Multi-field spaces

    X = MultiFieldFESpace([U, U_p1, U_p2, U_p3])   # (u, p₁, p₂, p₃)


    op = FEOperator(res, jac, X, Y)

    println("\n=== Step $step / $nsteps – disp_y = $disp_y ===\n")
    xh = FEFunction(X, x0)
    xh, cache = solve!(xh, solver, op, cache)

    uh, ph1, ph2, ph3 = xh

    # ----- VTK output (includes material_id) -----
    mkpath("TMC_C_shape_3D_approx")
    writevtk(
        Ω,
        "TMC_C_shape_3D_approx/results_$(lpad(step,3,'0'))";
        cellfields = [
            "uh"         => uh,
            "ph₁"        => ph1,
            "ph₂"        => ph2,
            "ph₃"        => ph3,
            "sigma"  => σ ∘ (ε(uh),material_field),
            "fskew₁" => Reg_fskew₁ ∘ (∇(uh), ph1, ∇(ph1)),
            "fskew₂" => Reg_fskew₂ ∘ (∇(uh), ph2, ∇(ph2)),
            "fskew₃" => Reg_fskew₃ ∘ (∇(uh), ph3, ∇(ph3)),
            "material_id"=> material_field
        ]
    )
    return get_free_dof_values(xh), cache
end

function run_all()
    disp_max = -0.3               # final vertical displacement (down)
    nsteps   = 60
    x0 = zeros(Float64, num_free_dofs(Y))
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
println("\nFinished – open folder 'TMC_C_shape_3D_approx' in Paraview")