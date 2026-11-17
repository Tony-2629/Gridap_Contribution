# --------------------------------------------------------------------
# Import packages
#---------------------------------------------------------------------

using Gridap
using Gridap.Fields
using Gridap.Geometry
using Gridap.CellData
using Gridap.FESpaces
using LineSearches: BackTracking
using Statistics


# Dimension of the model
D = 2

#-----------------------------------------------------------------------
# Geometry - C_shape body
#-----------------------------------------------------------------------
L, H = 1.24, 0.6                    #length, and height
h_bot = 0.2*H                       # yellow bottom thickness
h_mid = 0.6*H                       # brown middle thickness
h_top = 0.2*H                       # yellow top thickness
d = L/30;

# Yellow (stiff)   → K₁, μ₁
γ  = 1.0e-5              # scaling parameter
K₁ = 5.0/3.0;  μ₁ = 5/14; λ₁ = 30/21;

# Brown  (soft)    → K₂, μ₂
K₂ = K₁*γ;  μ₂ = μ₁*γ; λ₂ = λ₁*γ; 

K = 5.0/3.0;  μ = 5/14; λ = 30/21;
# Regularisation parameters

#-----------------------------------------------------------------------
# Discretisation
#-----------------------------------------------------------------------
Nx, Ny = 31, 15                     # cells along x and y within C-shape
domain = (0, L, 0, H)
partition = (Nx,Ny)
model = CartesianDiscreteModel(domain, partition)
Ω = Triangulation(model)

# Tag cells by centroid coordinate - mark cells
cell_points = get_cell_coordinates(Ω)          # 4 corner points per cell
cell_centroids = lazy_map(mean, cell_points)   # ← one centroid per cell
centroid_y = get_array(lazy_map(p -> p[2], cell_centroids))
# Material IDs (1 = solid, 2 = third medium)
material_id = zeros(Int, num_cells(model))
for (i, q) in enumerate(cell_centroids)
    (x,y) = q 
    if ((y ≤ h_bot || y ≥ (h_bot + h_mid) || x ≤ 0.2*H) && x ≤ 1.2)    # yellow layers
        material_id[i] = 1
    else                                                 # brown middle
        material_id[i] = 2
    end
end
material_field = CellField(material_id, Ω);   # for VTK

#-----------------------------------------------------------------------
# Find node by coordinate (tolerance = 10e-6)
#-----------------------------------------------------------------------

tol = 1e-6
tip_point = Point(1.2, 0.6)                 # (x,y) coordinate of node

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
# 6. Constitutive law – Neo-Hookean (per cell)
# -------------------------------------------------
F(∇u) = one(∇u) + ∇u'
C(∇u) = F(∇u)' ⋅ F(∇u)
J(∇u) = sqrt(det(C(∇u)))

# Virtual Green-Lagrange strain increment
δE(∇v, ∇u) = 0.5 * (∇v ⋅ F(∇u) + F(∇u)' ⋅ ∇v')

# Second Piola-Kirchhoff stress
function S(∇u, tag)
    λ, μ = (tag == 1) ? (λ₁, μ₁) : (λ₂, μ₂)
    Cinv = inv(C(∇u))
    return μ*(one(∇u) - Cinv) + λ*log(J(∇u))*Cinv
end

# Element type
order = 1
degree = 2*order+2
dΩ = Measure(Ω, degree)

#-----------------------------------------------------------------------
# Boundary condition tags
#-----------------------------------------------------------------------

labels = get_face_labeling(model)
entity = num_entities(labels)+1
labels.d_to_dface_to_entity[1][tip_node_id_global] = entity
add_tag!(labels,"Loads",[entity])
add_tag_from_tags!(labels, "Fixed", [1, 3, 7])   # left edge + corners

# Displacemence and deformation gradient averaging at the centroid of the element
reffe_u = ReferenceFE(lagrangian, VectorValue{D,Float64}, order)
V = TestFESpace(model, reffe_u, conformity=:H1,
                dirichlet_tags=["Fixed", "Loads"])
reffe_θ = ReferenceFE(lagrangian, TensorValue{D,D,Float64},order)
V_θ = TestFESpace(model, reffe_θ, conformity=:H1)

#----------------------------------------------------
# Weak form (based on the principle of virtual work)
#----------------------------------------------------
function ℝ((u,θ),(v,w))
    ∇u = ∇(u)
    ∇v = ∇(v)

    # Compose F and δF pointwise
    F_val = F∘(∇u)
    κ = 5.0e-3 # Pa
    α_r = 1.0e-9
    based_material = ∫((δE ∘ (∇(v), ∇(u))) ⊙ (S ∘ (∇(u), material_field)))dΩ
    
    # deformation gradient averaging at the centroid of the element
    # Ψᵣ = 0.5 * κ * ||F - Θ||² 
    # The first variation of the regularization term
    δW_r = ∫( κ * (F_val - θ) ⊙ (∇v) + κ * (θ - F_val) ⊙ w )dΩ
    δW_regularization = ∫(α_r*(∇(θ) ⊙ ∇(w)))dΩ

    return based_material + δW_r + δW_regularization
end

#----------------------------------------------------
# Solver - Newton solver with line search
#----------------------------------------------------

nls = NLSolver(show_trace=true, method=:newton,
    linesearch= BackTracking(), iterations = 200)

solver = FESolver(nls)

# Setup adaptive loading strategy or constant loading step

function run_step_Θ(x0, load_y, step, cache, fesolver, regularization_term)
    g_clamped = VectorValue(0.0, 0.0)
    g_loaded  = VectorValue(0.0, load_y)        # Vertical prescribed displacement

    U = TrialFESpace(V, [g_clamped, g_loaded])
    Θ = TrialFESpace(V_θ)

    # Multi-field spaces    
    X = MultiFieldFESpace([U, Θ])   # (u, θ)
    Y = MultiFieldFESpace([V, V_θ])   # (v, w)    

    op = FEOperator(ℝ, X, Y)
    xh = FEFunction(X, x0)
    # Safe solver execution tracking
    local solver_failed = false    

    try
        xh, cache = solve!(xh, solver, op, cache)
    catch
        solver_failed = true
    end
    uh, θh = xh
    r_vector = Gridap.FESpaces.residual(op, xh)
    r_norm = norm(r_vector)
    # Define your own success criteria
    # Usually, a residual < 1e-6 is excellent for non-linear mechanics
    if r_norm > 1e-6 || solver_failed
        println("!!! Step $step DIVERGED")
        return nothing, nothing, false
    end
    
    # ----- VTK output (includes material_id) -----
    data = ["u" => uh, "material_id"=> material_field]
    mkpath(regularization_term)
    writevtk(
        Ω,"$regularization_term/step_$(lpad(step,3,'0'))";
        cellfields = data
    )    
    return get_free_dof_values(xh), cache, true
end

function run_adaptive_loading_Θ(fesolver, regularization_term, target_displacement, adaptive_load, initial_increment, min_increment, max_increment)
    max_load = target_displacement
    last_load = 0.0
    dload = initial_increment
    min_dload = min_increment
    max_dload = max_increment
    load = last_load + dload
    num_iteration = 0
    load_step = 1
    U_init = TrialFESpace(V, [VectorValue(0.0, 0.0), VectorValue(0.0, 0.0)])
    Θ_init = TrialFESpace(V_θ)
    X_initial = MultiFieldFESpace([U_init, Θ_init])
    #Uq_init = TrialFESpace(V_q)
    I_tensor = one(TensorValue{D, D, Float64})
    u_init = zero(U_init)
    θ_init = interpolate(I_tensor, Θ_init)
    u_0 = VectorValue(0.0, 0.0)
    xh_init = interpolate([u_0, I_tensor], X_initial)
    x_converged = get_free_dof_values(xh_init)
    cache = nothing

    while abs(load) < abs(max_load)
        increment = sign(max_load)*min(dload, abs(max_load - load))
        trial_load = load + increment

        println("\n>>> LOAD STEP: u_y $trial_load (dload: $increment)")
        
        u_new, new_cache, success = run_step_Θ(x_converged, trial_load, load_step, cache, fesolver,regularization_term)
        if success
            # Success: Move state forward
            load_step += 1
            x_converged = u_new
            cache = new_cache
            load = trial_load
            if abs(load) < abs(max_load)
                dload = (dload >= max_dload) ? dload : min(dload * 1.1, abs(max_load - load))
            end          
        else
            # Failure: Backtrack and cut load step
            println("!!! DIVERGENCE: Cutting load step size from $dload to $(dload/2)")
            dload /= 2
            if dload < min_dload
                error("Simulation aborted: Step size dropped below threshold.")
            end
        end    
    end
    return
end
run_adaptive_loading_Θ(solver, "gradient_based_regularization_out", -0.6, true, 0.004, 0.002, 0.005)