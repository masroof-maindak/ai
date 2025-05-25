# Neural Networks

- [x] Lab 7 is well-annotated and ideal for studying.
- [x] Asg 6 was created after lab 7 and is the dumbed down version of it
- [ ] [ **wontfix** ] Lab 8 adds nothing to Lab 7 other than regular dropout, hence the AI slop
- [ ] [ **wontfix** ] Lab 9 adds nothing to Lab 7 other than extending it from a binary classifier to a generic classifier. Shouldn't require much modification other than changing the activation function and other dataset-specific dictionary maintenance -- Softmax?

## Questions

- [ ] How is the derivative of the 'loss' (or equivalent) w.r.t AL (for any layer prior to the final layer) equal to `delta2 @ self.W[L+1].T`
- [x] Why is MSE + Sigmoid sub-optimal?
    - Dr Usama: With BCE, intermediate pre-activations are more likely to be well-behaved. *LOOK INTO THIS*. With MSE, if accuracy doesn't increase, it's likely that you've encountered the vanishing gradient problem w.r.t to the Sigmoid activation
    - It's not that MSE and Sigmoid together are sub-optimal, it's that MSE in general is unsuitable for classification tasks.
        - For one, MSE wouldn't differentiate between the distance from 0.49 to 0.51, and 0.89 to 0.91, but in a classification problem where the label is True, those two changes do not have the same effect. So giving them both the same reward or punishment is not a good strategy.
        - Furthermore, when the model does predict probabilities close to 0 or 1, the cost function's gradient (derivative) becomes very small, making it harder to correct mistakes, thereby slowing down learning. Conversely, When the true label is y=1, cross-entropy encourages the prediction `y_hat` to be as close to 1 as possible by minimizing `−log(y_hat)`.
        - Lastly, cross-entropy promises a convex loss surface, whereas MSE would create a non-convex surface. This translates to the presence of multiple local minima, making it difficult for the gradient descent algorithm to find the global optimum.
        - Sources: 
            - [Deeplearning.ai - paulinpaloalto](https://community.deeplearning.ai/t/mse-cost-function/23349/2)
            - [deeplearning.ai - nadtriana](https://community.deeplearning.ai/t/use-of-squared-error-with-sigmoid-and-applying-gradient-descent/700239/2)
- [ ] How does the derivative of BCE+Sigmoid/CE+Softmax applied on a layer during back-prop simplify to such an elegant expression?

---

- Softmax - Sigmoid for multi-label - also exhibits Vanishing Gradient; read more @ `deeplearningbook.org`
- Computational graph
- Automated differentiation
- Reverse Mode
- Loss is a function over weights and biases, i.e they are they parameters
