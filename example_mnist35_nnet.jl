# Load X and y variable
using JLD, Printf
data = load("mnist35.jld")
(X, y, Xtest, ytest) = (data["X"], data["y"], data["Xtest"], data["ytest"])
y[y.==2] .= -1
ytest[ytest.==2] .= -1
(n, d) = size(X)

## Fit logistic regression model before NN
include("logreg.jl")
model = logReg12(X, y)
res = y - model.predict(X)

# Choose network structure and randomly initialize weights
include("NeuralNet.jl")
nHidden = [3]
nParams = NeuralNet_nParams(d, nHidden)
w = randn(nParams, 1)
w[end-d+1:end] = model.w

# Train with stochastic gradient
maxIter = 10000
stepSize(t) = 1e-1 / sqrt(t)
for t in 1:maxIter

    # The stochastic gradient update:
    batch_size = round(n * 0.2)

    f = 0
    g = zeros(nParams, 1)
    # The stochastic gradient update:
    for iter in 1:batch_size
        i = rand(1:n)
        (f_i, g_i) = NeuralNet_backprop(w, X[i, :], y[i], nHidden)
        g = g .+ g_i
        f = f + f_i
    end

    g = g ./ batch_size
    f = f ./ batch_size

    global w = w - stepSize(t) * g

    # Every few iterations, plot the data/model:
    if (mod(t - 1, round(maxIter / 50)) == 0)
        yh = sign.(NeuralNet_predict(w, Xtest, nHidden))
        errorRate = sum(yh .!= ytest) / size(Xtest, 1)
        @printf("Training iteration = %d, error rate = %.2f\n", t - 1, errorRate)
    end
end
