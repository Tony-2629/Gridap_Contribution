using Gridap
using Gridap.Geometry
using Gridap.TensorValues

# -------------------------------------------------
# 2. Geometry – three rectangles (union = full domain)
# -------------------------------------------------
L, H = 1.025, 0.5                     # total size
h_bot = 0.2*H                       # yellow bottom thickness
h_mid = 0.6*H                       # brown middle thickness
h_top = 0.2*H                       # yellow top thickness
domain = (0, L, 0, H)
partition = (41, 20)
model = CartesianDiscreteModel(domain, partition)


Ω = Triangulation(model)
# -------------------------------------------------
# 2. Tag cells by centroid Y-coordinate
# -------------------------------------------------

cell_points = get_cell_coordinates(Ω)          # 4 corner points per cell
cell_centroids = lazy_map(mean, cell_points)   # ← one centroid per cell

# Extract Y-coordinate of each centroid
centroid_y = get_array(lazy_map(p -> p[2], cell_centroids))


# Material IDs (1 = yellow, 2 = brown)
material_id = zeros(Int, num_cells(model))
for (i, q) in enumerate(cell_centroids)
    (x,y) = q 
    if ((y ≤ h_bot || y ≥ (h_bot + h_mid) || x ≤ 0.2*H) && x ≤ 1.0)    # yellow layers
        material_id[i] = 1
    else                                                               # brown middle
        material_id[i] = 2
    end
end
material_field = CellField(material_id, Ω)                             # for VTK

# -------------------------------------------------
# 3. Material parameters
# -------------------------------------------------
λ₁, μ₁ = 100.0, 1.0          # yellow – stiff
λ₂, μ₂ = λ₁*1e-5, μ₁*1e-5     # brown  – soft

# -------------------------------------------------
# 4. σ(ε, tag) – your function
# -------------------------------------------------
function σ(ε, tag::Int)
    if tag == 1
        return λ₁*tr(ε)*one(ε) + 2*μ₁*ε
    else
        return λ₂*tr(ε)*one(ε) + 2*μ₂*ε
    end
end

# -------------------------------------------------
# FIND NODE BY COORDINATE (tolerance = 1e-6)
# -------------------------------------------------
tol = 1e-6
tip_point = Point(1.0, 0.50)  # (x,y) of the node

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

println("Prescribed displacement at node $tip_node_id_global: (1.0, 0.5)")

# -------------------------------------------------
# 2. Labels
# -------------------------------------------------
labels = get_face_labeling(model)
entity = num_entities(labels)+1
labels.d_to_dface_to_entity[1][tip_node_id_global] = entity
add_tag!(labels,"Tip",[entity])
add_tag_from_tags!(labels, "Dirichlet", [1, 3, 7])   # left edge → clamped
#add_tag_from_tags!(labels, "Tip",       [4])        # right tip → prescribed




# -------------------------------------------------
# 3. Integration
# -------------------------------------------------
order = 1
degree = 2*order
dΩ = Measure(Ω, degree)
# -------------------------------------------------
# 4. FE Space
# -------------------------------------------------

reffe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
V = TestFESpace(model, reffe_u, conformity=:H1, dirichlet_tags=["Dirichlet", "Tip"])
g_clamped(x) = VectorValue(0.0, 0.0)
g_tip(x)     = VectorValue(0.0, -0.5)
U = TrialFESpace(V, [g_clamped, g_tip])

# Regularisation field p (scalar)
reffe_p = ReferenceFE(lagrangian, Float64, order)
V_p = TestFESpace(model, reffe_p, conformity=:H1)
U_p = TrialFESpace(V_p)


# Multi-field spaces
Y = MultiFieldFESpace([V, V_p])   # (v, ξ)
X = MultiFieldFESpace([U, U_p])   # (u, p)

# -------------------------------------------------
# 6. Enhanced gradient regularisation (only on brown, tag == 2)
# -------------------------------------------------
# Regularisation parameters
γ  = 1.0e-5              # weight
β₁ = 10000.0             # (α - p)^2 coefficient
α_r = 100.0               # ∇p⋅∇p length scale

# Shear measure α = (F12 - F21)/(F11 + F22) ≈ (∂u1/∂x2 - ∂u2/∂x1)/(∂u1/∂x1 + ∂u2/∂x2 + 2)

# Deformation gradient: F(∇u) = I + (∇u)'
F(∇u) = one(∇u) + ∇u'
# α(∇u) = (F12 - F21)/(F11 + F22)
function α(∇u::TensorValue{2,2,Float64})
F_val = F(∇u)
    a = F_val[1,2] - F_val[2,1]
    d = F_val[1,1] + F_val[2,2] + 1e-12
    return a / d
end

# G(∇u) = ∂α/∂(∇u)
function G(∇u::TensorValue{2,2,Float64})
    F_val = F(∇u)
    a = F_val[1,2] - F_val[2,1]
    b = (F_val[1,1] + F_val[2,2])
    d = (F_val[1,1] + F_val[2,2] + 1e-12)^2    
    return ∇u #TensorValue(-a/b, b/b, -b/b, a/b )
end


# Mask: only brown (tag == 2)
reg_mask = CellField(material_id .== 2, Ω)

function norm_u(u)
    return norm(u) + 1e-12
end

function grad_u(u)
    grad = β₁ * (norm_u(u) - p)
    return grad
end



# -------------------------------------------------
# 6. Weak Form (NO LOAD)
# -------------------------------------------------
function a((u,p),(v,ξ))
    elas = ∫( ε(v) ⊙ (σ ∘ (ε(u), material_field)))dΩ
    #reg_u = ∫( β₁*( (α ∘ ∇(u)) - p )*(G∘∇(u)⊙∇(v))*reg_mask)dΩ
    reg_u = ∫( (norm_u ∘ u - p) * (u ⋅ v) * reg_mask )dΩ
    reg_p = ∫( β₁ * (norm_u ∘ u - p) * ξ + α_r * ∇(p) ⋅ ∇(ξ) * reg_mask )dΩ
    return elas + reg_p #+ reg_u
end

l((v,ξ)) = 0

# -------------------------------------------------
# 7. Solve
# -------------------------------------------------
op = AffineFEOperator(a, l, X, Y)
xh = solve(op)
uh, ph = xh

# -------------------------------------------------
# 8. Export
# -------------------------------------------------
writevtk(Ω, "cshape_reg",
         cellfields=[
             "uh"     => uh,
             "sigma"  => σ ∘ (ε(uh),material_field),
             "material_id"=> material_field
         ])
# -------------------------------------------------
# 12. TYPE CHECK + EXPORT G
# -------------------------------------------------

println("Saved: cshape_reg.vtk")
println("Tip displacement: u_y = -0.4")