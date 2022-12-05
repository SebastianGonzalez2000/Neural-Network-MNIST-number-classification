# We use nHidden as a vector, containing the number of hidden units in each layer
# Definitely not the most efficient implementation!

# Function that returns total number of parameters
function NeuralNet_nParams(d, nHidden)

    # Connections from inputs to first hidden layer
    nParams = (d+1) * nHidden[1]

    # Connections between hidden layers
    for h in 2:length(nHidden)
        nParams += (nHidden[h-1]+1) * nHidden[h]
    end

    # Connections from last hidden layer to output (+d for skip connections)
    nParams += nHidden[end] + d+1

end

# Compute squared error and gradient
# for a single training example (x,y)
# (x is assumed to be a column-vector)
function NeuralNet_backprop(bigW, x, y, nHidden)
    d = length(x)
    nLayers = length(nHidden)

    #### Reshape 'bigW' into vectors/matrices
    # - This is not a really elegant way to do things
    # if you want to be really efficient, but for the course
    # it is nice abraction
    W1 = reshape(bigW[1:nHidden[1]*(d+1)], nHidden[1], d+1)
    ind = nHidden[1] * (d+1)
    Wm = Array{Any}(undef, nLayers - 1)
    for layer in 2:nLayers
        Wm[layer-1] = reshape(bigW[ind+1:ind+nHidden[layer]*(nHidden[layer-1]+1)], nHidden[layer], nHidden[layer-1]+1)
        ind += nHidden[layer] * (nHidden[layer-1]+1)
    end
    v = bigW[ind+1:end]

    #### Define activation function and its derivative
    h(z) = tanh.(z)
    dh(z) = (sech.(z)) .^ 2

    #### Forward propagation
    z = Array{Any}(undef, nLayers)
    z[1] = W1 * [x;1]
    for layer in 2:nLayers
        z[layer] = Wm[layer-1] * [h(z[layer-1]);1]
    end
    yhat = v' * [h(z[end]); x;1]

    r = log(1 + exp(-y * yhat))
    f = r

    #### Backpropagation (the below could be replaced by AD)
    dr = (-y * exp(-y * yhat) / (1 + exp(-y * yhat)))
    err = dr
    lambda = 0.01

    # Output weights
    Gout = err * [h(z[end]); x; 1] + lambda * v

    Gm = Array{Any}(undef, nLayers - 1)
    if nLayers > 1
        # Last Layer of Hidden Weights, v[1:end-d] to ignore skip connections
        backprop = err * (dh(z[end]) .* v[1:end-d-1])
        Gm[end] = backprop * [h(z[end-1])' 1] + lambda * Wm[end]

        # Other Hidden Layers
        for layer in nLayers-2:-1:1
            backprop = (Wm[layer+1][:,1:end-1]' * backprop) .* dh(z[layer+1])
            Gm[layer] = backprop * [h(z[layer])' 1] + lambda * Wm[layer]
        end

        # Input Weights
        backprop = (Wm[1][:,1:end-1]' * backprop) .* dh(z[1])
        G1 = backprop * [x' 1] + lambda * W1
    else
        # Input weights, v[1:end-d] to ignore skip connections
        G1 = err * (dh(z[1]) .* v[1:end-d-1]) * [x' 1] + lambda * W1
    end

    #### Put gradients into vector
    g = zeros(size(bigW))
    g[1:nHidden[1]*(d+1)] = G1
    ind = nHidden[1] * (d+1)
    for layer in 2:nLayers
        g[ind+1:ind+nHidden[layer]*(nHidden[layer-1]+1)] = Gm[layer-1]
        ind += nHidden[layer] * (nHidden[layer-1]+1)
    end
    g[ind+1:end] = Gout

    return (f, g)
end

# Computes predictions for a set of examples X
function NeuralNet_predict(bigW, Xhat, nHidden)
    (t, d) = size(Xhat)
    nLayers = length(nHidden)

    #### Reshape 'bigW' into vectors/matrices
    W1 = reshape(bigW[1:nHidden[1]*(d+1)], nHidden[1], d+1)
    ind = nHidden[1] * (d+1)
    Wm = Array{Any}(undef, nLayers - 1)
    for layer in 2:nLayers
        Wm[layer-1] = reshape(bigW[ind+1:ind+nHidden[layer]*(nHidden[layer-1]+1)], nHidden[layer], nHidden[layer-1]+1)
        ind += nHidden[layer] * (nHidden[layer-1]+1)
    end
    v = bigW[ind+1:end]

    #### Define activation function and its derivative
    h(z) = tanh.(z)
    dh(z) = (sech.(z)) .^ 2


    #### Forward propagation on each example to make predictions
    yhat = zeros(t, 1)
    for i in 1:t
        # Forward propagation
        z = Array{Any}(undef, 1nLayers)
        z[1] = W1 * [Xhat[i, :]; 1]
        for layer in 2:nLayers
            z[layer] = Wm[layer-1] * [h(z[layer-1]);1]
        end
        yhat[i] = v' * [h(z[end]); Xhat[i, :]; 1]
    end
    return yhat
end

