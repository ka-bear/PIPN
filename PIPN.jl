using Flux

# Define shared MLP layers
shared_mlp1 = Flux.Chain(
    Dense(2, 64, tanh),  # Each data point has 2 features
    Dense(64, 64, tanh)
)

shared_mlp2 = Flux.Chain(
    Dense(64, 128, tanh),
    Dense(128, 1024, tanh)
)

# Define the rest of the MLP for after max pooling
after_pool_mlp = Flux.Chain(
    Dense(1024, 512, tanh),
    Dense(512, 256, tanh),
    Dense(256, 128, tanh)
)

# Define final layer for prediction
final_layer = Dense(128, 2)

function apply_shared_mlp(shared_mlp, points)
    return [shared_mlp(vec) for vec in eachcol(points)]
end


function aggregate_global_feature(points)
    return maximum(points, dims=2)
end

function model(points)
    points = apply_shared_mlp(shared_mlp1, points)
    points = apply_shared_mlp(shared_mlp2, points)
    global_feature = aggregate_global_feature(points)

    # Ensure global_feature is repeated for each point
    global_feature_repeated = repeat(global_feature, 1, size(points, 2))

    # Concatenate each point with the global feature
    points = [vcat(points[:, i], global_feature_repeated[:, i]) for i in 1:size(points, 2)]
    points = apply_shared_mlp(after_pool_mlp, hcat(points...))
    return final_layer(points)
end

simple_mlp = Flux.Chain(
    Dense(2, 64, tanh),
    Dense(64, 2)
)

# Simple model function
function simple_model(points)
    return simple_mlp(points)
end

using Flux

# Define shared MLP layers (applied to each point)
shared_mlp1 = Flux.Chain(
    Dense(2, 64, tanh),
    Dense(64, 128, tanh)
)

# Aggregate global feature
function aggregate_global_feature(points)
    return maximum(points, dims=2)
end

# Define MLPs after aggregation that can handle the increased feature size
after_pool_mlp = Flux.Chain(
    Dense(256, 128, tanh),  
    Dense(128, 64, tanh)
)

# Final layer for prediction
final_layer = Dense(64, 2)

# Apply shared MLP to each point
function apply_shared_mlp(shared_mlp, points)
    return map(x -> shared_mlp(x), eachcol(points))
end

# PointNet model function
function pointnet_model(points)
    # Apply shared MLPs
    point_features = apply_shared_mlp(shared_mlp1, points)
    point_features = hcat(point_features...)

    # Aggregate to global feature
    global_feature = aggregate_global_feature(point_features)

    # Flatten the global feature to match point features dimensions
    global_feature_flattened = repeat(reshape(global_feature, :), 1, size(point_features, 2))

    # Combine point features with global feature
    combined_features = vcat(point_features, global_feature_flattened)

    # Apply additional MLPs and final layer
    combined_features = after_pool_mlp(combined_features)
    return final_layer(combined_features)
end

# Loss function
loss_fn(y_true, y_pred) = Flux.mse(y_true, y_pred)

# Optimizer
optimizer = Flux.ADAM(0.001, (0.9, 0.999))

num_epochs = 100
batch_size = 32

num_gross = 6000

x_fire = fill!(similar(zeros(Float64, num_gross)), 100)
y_fire = fill!(similar(zeros(Float64, num_gross)), 100)
u_fire = fill!(similar(zeros(Float64, num_gross)), 100)
v_fire = fill!(similar(zeros(Float64, num_gross)), 100)
T_fire = fill!(similar(zeros(Float64, num_gross)), 100)
dTdx_fire = fill!(similar(zeros(Float64, num_gross)), 100)
dTdy_fire = fill!(similar(zeros(Float64, num_gross)), 100)

function readFire(number, name)
    coord = 1
    open("C:/Users/admin/PhysicsInformedPointNetElasticity/data/" * name * "x" * string(number) * ".txt") do f
        for line in eachline(f)
            x_fire[coord] = parse(Float64, split(line)[1])
            coord += 1
        end
    end

    coord = 1
    open("C:/Users/admin/PhysicsInformedPointNetElasticity/data/" * name * "y" * string(number) * ".txt") do f
        for line in eachline(f)
            y_fire[coord] = parse(Float64, split(line)[1])
            coord += 1
        end
    end

    coord = 1
    open("C:/Users/admin/PhysicsInformedPointNetElasticity/data/" * name * "u" * string(number) * ".txt") do f
        for line in eachline(f)
            u_fire[coord] = parse(Float64, split(line)[1]) * 1
            coord += 1
        end
    end

    coord = 1
    open("C:/Users/admin/PhysicsInformedPointNetElasticity/data/" * name * "v" * string(number) * ".txt") do f
        for line in eachline(f)
            v_fire[coord] = parse(Float64, split(line)[1]) * 1
            coord += 1
        end
    end

    coord = 1
    open("C:/Users/admin/PhysicsInformedPointNetElasticity/data/" * name * "dTdx" * string(number) * ".txt") do f
        for line in eachline(f)
            dTdx_fire[coord] = parse(Float64, split(line)[1]) / 1
            coord += 1
        end
    end

    coord = 1
    open("C:/Users/admin/PhysicsInformedPointNetElasticity/data/" * name * "dTdy" * string(number) * ".txt") do f
        for line in eachline(f)
            dTdy_fire[coord] = parse(Float64, split(line)[1]) / 1
            coord += 1
        end
    end

    coord = 1
    open("C:/Users/admin/PhysicsInformedPointNetElasticity/data/" * name * "T" * string(number) * ".txt") do f
        for line in eachline(f)
            T_fire[coord] = parse(Float64, split(line)[1]) / 1
            coord += 1
        end
    end
end


data_square = 39
data_pentagon = 30
data_hexagon = 29
data_heptagon = 25
data_octagon = 22
data_nonagon = 19
data = data_square + data_pentagon + data_hexagon + data_heptagon + data_octagon + data_nonagon

x_fire_load = fill!(similar(zeros(Float64, data, num_gross)), 100)
y_fire_load = fill!(similar(zeros(Float64, data, num_gross)), 100)
u_fire_load = fill!(similar(zeros(Float64, data, num_gross)), 100)
v_fire_load = fill!(similar(zeros(Float64, data, num_gross)), 100)
T_fire_load = fill!(similar(zeros(Float64, data, num_gross)), 100)
dTdx_fire_load = fill!(similar(zeros(Float64, data, num_gross)), 100)
dTdy_fire_load = fill!(similar(zeros(Float64, data, num_gross)), 100)

list_data = [data_square, data_pentagon, data_hexagon, data_heptagon, data_octagon, data_nonagon]
list_name = ["square", "pentagon", "hexagon", "heptagon", "octagon", "nanogan"]

global counter

counter = 1

for (k, list_data_k) in enumerate(list_data)
    for i in 1:list_data_k
        readFire(2 * i + 1, list_name[k])
        global counter
        for j in 1:num_gross
            global counter
            x_fire_load[counter, j] = x_fire[j]
            y_fire_load[counter, j] = y_fire[j]
            u_fire_load[counter, j] = u_fire[j]
            v_fire_load[counter, j] = v_fire[j]
            T_fire_load[counter, j] = T_fire[j]
            dTdx_fire_load[counter, j] = dTdx_fire[j]
            dTdy_fire_load[counter, j] = dTdy_fire[j]
        end
        counter += 1
    end
end

using Flux

num_epochs = 100
batch_size = 32
function reshape_to_2_64(matrix)
    original_size = size(matrix, 2)
    if original_size > 64
        # Truncate the matrix
        return matrix[:, 1:64]
    else
        # Pad the matrix with zeros
        padding = zeros(size(matrix, 1), 64 - original_size)
        return hcat(matrix, padding)
    end
end

# Combine inputs into one matrix and targets into another where each column represents a data point
inputs_matrix = hcat(x_fire_load, y_fire_load) 
targets_matrix = hcat(u_fire_load, v_fire_load) 

# DataLoader
data = [(inputs_matrix[:, i], targets_matrix[:, i]) for i in 1:size(inputs_matrix, 2)]
final_dataset = Flux.Data.DataLoader(data, batchsize=batch_size, shuffle=true)

# Training loop
num_epochs = 100
batch_size = 32

for epoch in 1:num_epochs
    println("Epoch: ", epoch)
    for (input, target) in final_dataset
        # Ensure the input and target are in the correct shape
        input_matrix = transpose(hcat(input...))
        target_matrix = transpose(hcat(target...))

        gs = Flux.gradient(Flux.params(shared_mlp1, after_pool_mlp, final_layer)) do
            predictions = pointnet_model(input_matrix)
            loss = loss_fn(target_matrix, predictions)
        end


        println("Loss: ", loss_fn(target_matrix, simple_model(input_matrix)))
        Flux.Optimise.update!(optimizer, Flux.params(shared_mlp1, after_pool_mlp, final_layer), gs)
    end
end

