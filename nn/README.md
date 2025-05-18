# Neural Networks

- [x] Lab 7 is well-annotated and ideal for studying.
- [x] Asg 6 was created after lab 7 and is the dumbed down version of it
- [ ] [ **wontfix** ] Lab 8 adds nothing to Lab 7 other than regular dropout, hence the AI slop
- [ ] [ **wontfix** ] Lab 9 adds nothing to Lab 7 other than extending it from a binary classifier to a generic classifier. Shouldn't require much modification other than changing the activation function and other dataset-specific dictionary maintenance -- Softmax?

## Questions

- [ ] How is the derivative of the 'loss' (or equivalent) w.r.t AL (for any layer prior to the final layer) equal to `delta2 @ self.W[L+1].T`
- [ ] Why is MSE + Sigmoid sub-optimal?
    - With BCE, intermediate pre-activations are more likely to be well-behaved. *LOOK INTO THIS*. With MSE, if accuracy doesn't increase, it's likely that you've encountered the vanishing gradient problem w.r.t to the Sigmoid activation
- [ ] How does the derivative of BCE + Sigmoid applied on a layer during back-prop simplify to such an elegant expression?

- Softmax - Sigmoid for multi-label - also exhibits Vanishing Gradient ; read more @ `deeplearningbook.org`
- Computational graph
- Automated differentiation
- Reverse Mode
- Loss is a function over weights and biases, i.e they are they parameters
