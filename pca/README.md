Goal w/ PCA: "If I pick an arbitrary direction v in my feature space, how much of my data's total variance is captured along this specific direction v?"

This begs the question, why are we after variance anwyay? Simply put, it's because the variance is used as a measure of information/strength. If your data points are widely spread out along a particular direction (axis/feature), it means the data takes on a diverse range of values in that direction. This spread often corresponds to meaningful differences or patterns between the data points.

So then how do we calculate the projected variance? `v^T C v`, where `C` is the covariance matrix and `v` is a (unit) direction vector. How does that work out? I don't know, I got a B+ in Linear Algebra.

It's still fairly straightforward though that the goal now is to find the **best** `v`. Luckily for us, this is exactly the same as solving for the Eigenvectors of `C`.