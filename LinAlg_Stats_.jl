## Listing 3.4 --> Statistics with Julia (p. 79, Springer, 2022)
# %%
using Distributions, Plots;
pyplot();

dist = TriangularDist(4, 6, 5)
N = 10^6
data = rand(dist, N)
yData = (data .- 5) .^ 2

println("Mean: ", mean(yData), " Variance: ", var(data))

p1 = histogram(data, xlabel="x", bins=80, normed=true, ylims=(0, 1.1))
p2 = histogram(yData, xlabel="y", bins=80, normed=true, ylims=(0, 15))
plot(p1, p2, ylabel="Proportion", size=(800, 400), legend=:none)
# %%
"""
The Triangular() function from Distributions is used to create a triangular distribution-type object with a mean of 5
and a symmetric shape over the bound [4, 6]. An array of N observations from the distribution is then generated
by applying the rand() function on the distribution `dist`. The observation in `data` and from them generates observations
for the new RV variable `y`. The values are stored in the array `yData`. Furthermore, the functions mean() and var() on the arrays
`yData` and `data`, respectively. The output that the mean of distribution `Y` is the same as the variance of `X`. The histograms are plotted
of the data in the arrays `data` and `yData`. The histogram on the left approximates the PDF of the triangular distribution,
while the histogram on the right approximates the distribution of the new variable `Y`. The distribution of `Y` is
seldom considered when evaluation the variance of `X`.
"""
# %%
# 3.5, p.83
# CDF from the Riemann sum of a PDF
using Plots, LaTeXStrings; pyplot()

f2(x) = (x < 0 ? x + 1 : 1 - x) * (abs(x) < 1 : 0)
a, b = -1.5, 1.5
dt = 0.01

F(x) = sum([f2(u)*dt for u in a:dt:x])

xGrid = a:dt:b
y = [F(u) for u in xGrid]
plot(xGrid, y, c = :blue, xlims = (a, b), ylims = (0, 1), xlabel = L"x", ylabel = L"F(x)", legend=:none)
