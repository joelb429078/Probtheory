Card 1
 Front: What is supervised learning?
 Back: A paradigm where the model learns a mapping ff from inputs xx to labels yy using a labeled dataset {(x(i),y(i))}\{(x^{(i)},y^{(i)})\}. Goal: predict y≈f(x)y\approx f(x) for new xx.

 Card 2
 Front: Give the hypothesis (model) for linear regression.
 Back:
h_\theta(x) = \theta_0 + \theta_1 x_1 + \cdots + \theta_n x_n = \theta^T x
(assuming x0=1x_0=1)

Card 3
 Front: What is the least-squares cost function J(θ)J(\theta) for linear regression?
 Back:
 J(\theta) = \frac{1}{2m}\sum_{i=1}^m\bigl(h_\theta(x^{(i)}) - y^{(i)}\bigr)^2

Card 4
 Front: Derive the gradient ∂J∂θj\frac{\partial J}{\partial \theta_j} for linear regression.
 Back:
∂J∂θj=1m∑i=1m(hθ(x(i))−y(i)) xj(i)\frac{\partial J}{\partial \theta_j} =\frac{1}{m}\sum_{i=1}^m\bigl(h_\theta(x^{(i)}) - y^{(i)}\bigr)\,x_j^{(i)}

Card 5
 Front: State the batch gradient descent update rule.
 Back:
\theta_j := \theta_j - \alpha \,\frac{1}{m}\sum_{i=1}^m\bigl(h_\theta(x^{(i)}) - y^{(i)}\bigr)x_j^{(i)}
where α\alpha is the learning rate.

Card 6
 Front: What problem arises if you use the least-squares cost for logistic regression?
 Back: It becomes non-convex in θ\theta, leading to multiple local minima and hard optimization.

Card 7
 Front: Write the sigmoid function \sigma(z).
 Back:
 \sigma(z) = \frac{1}{1 + e^{-z}}

Card 8
 Front: What is the hypothesis for logistic regression?
 Back:
h_\theta(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}
Interpreted as P(y=1\mid x;\theta).

Card 9
 Front: Give the cross-entropy (log) cost function for logistic regression.
 Back:
  J(\theta) = -\frac{1}{m}\sum_{i=1}^m\Bigl[y^{(i)}\log h_\theta(x^{(i)}) + (1 - y^{(i)})\log\bigl(1-h_\theta(x^{(i)})\bigr)\Bigr].

Card 10
 Front: What is the gradient of the logistic cost w.r.t.\ θj\theta_j?
 Back:
\frac{\partial J}{\partial \theta_j} =\frac{1}{m}\sum_{i=1}^m\bigl(h_\theta(x^{(i)}) - y^{(i)}\bigr)x_j^{(i)}.

Card 11
 Front: Define k-Nearest Neighbours (k-NN).
 Back: A non-parametric method that, to predict a query point, finds the kk closest training examples (by some distance), then:
Classification: majority vote of their labels
Regression: average of their targets

Card 12
 Front: Give the formula for Euclidean distance between points pp and qq.
 Back:
d(p,q) = \sqrt{\sum_{i=1}^n (p_i - q_i)^2}.

Card 13
 Front: What is the Manhattan distance?
 Back:
d(p,q) = \sum_{i=1}^n \lvert p_i - q_i\rvert.

Card 14
 Front: State one advantage and one disadvantage of k-NN.
 Back:
 Advantage: Simple, no training phase.
 Disadvantage: Slow at prediction time; sensitive to feature scaling and choice of k.

Card 15
 Front: What does a maximal margin classifier do?
 Back: Finds the hyperplane w^T x - b = 0 that separates two classes with the largest margin to the nearest points (support vectors).

Card 16
 Front: Write the decision rule (hypothesis) for a maximal margin classifier.
 Back:
    h_{w,b}(x) = \mathrm{sign}(w^T x - b).

Card 17
 Front: What are the margin constraints for linearly separable SVM?
 Back: For each (x^{(i)},y^{(i)}) with y^{(i)}\in\{-1,+1\}:
              y^{(i)}\bigl(w^T x^{(i)} - b\bigr)\ge 1.
              
Card 18
 Front: What objective does the hard-margin SVM optimize?
 Back:
 \min_{w,b}\;\frac{1}{2}\|w\|^2\quad\text{s.t. }y^{(i)}(w^T x^{(i)} - b)\ge1.




