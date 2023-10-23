# Boko Haram Model Fitting

There are several difficulties in fitting a model to the Boko Haram terrorist attack data:

1. There are many duplicate locations and times.

2. The data contains both endogenous and exogeneous inhomogeneity.


The first is at least superficially taken care of by adding random noise to the location data. Some of these duplicate locations are a result of repeated attacks, but most of them are a result of the granularity of the data. The location for an event that took place in a city if the exact location is not known is set to a predefined location at the center of the city.

I proposed to solve the second problem by two different methods:

- Stochastic Declustering Hawkes model

- Cox Hawkes model

These models operate in a similar manner. The background rate is inhomogeneous and estimated from the data, while the triggering mechanism is set to a parametric form.

Several considerations were similar in both cases. First, do you treat the background time as constant? Analysis showed that the background rate was not constant with time. In fact, political changes in Nigeria and the countries surrounding it dramatically affected the rate of attack. Therefore, the time and location were both estimated non-parametrically, however they were assumed to be independent in the estimate. 

$\mu(s,t) = f(t)g(s)$

Second, how do you parameterize the trigger function to get an accurate interpretable estimate. Again, time and location are assumed to be independent. We use the exponential pdf for time, the normal pdf for location, and another parameter for reproduction rate.


Finally, and perhaps most importantly, how can we incorporate covariates in the estimation of background and trigger intensities? In the model with no covariates, there were areas of high intensity that were underestimated in part because of the estimate was a kernel method. This could be remedied by adding covariates. Covariates for the trigger function are usually associated with the event rather than the location of the event. They are straight forward to add to the trigger function. Multiply a regression term to the trigger function as follows (See [(Adelfio 2015)](https://link.springer.com/article/10.1007/s10260-020-00543-5) or Section 6 of [(Chaing 2022)](https://scholarworks.iupui.edu/items/24e1401b-7559-4f25-8d08-fbd7661458a1)).

\begin{equation}
    g(t-t_i,s-s_i,m_i) = r_0 j(m_i) f(t-t_i) h(s-s_i)
\end{equation}

We require the trigger function to be positive, so $j$ must be a positive function. Often it is $j(m) = exp(m\cdot w)$ or $j(m_i)=1 + m\cdot w$ if the covariates are positive [(Li 2021)](https://onlinelibrary.wiley.com/doi/full/10.1002/env.2697). A wald test for significance can help select predictive variables. In our case, none of covariates we tried (civilian opponent in conflict, and number of deaths resulting from the conflict) were significant. 

When it comes to adding covariates to the background, it is slightly more complicated. A non-parametric estimate using a high dimensional dataset can severely overfit. Therefore, it is necessary to use some parametric form.


$\mu(s,t,x) = f(t)g(s) exp(x \cdot w)$


$\mu(s,t,x) = f(t)g(s,x \cdot w)$


I conducted initial analysis to determine which covariates were most predictive. In this setup, I aggregated the event counts on a grid with their associated coraviate values. I then conducted Poisson regression to select the best variables.

Further references:

- [Terrorism Spatiotemporal Hawkes](https://arxiv.org/pdf/2202.12346.pdf) with flexible trigger.

- [stelfi R package](https://cran.r-project.org/web/packages/stelfi/stelfi.pdf) with [documentation](https://cmjt.github.io/stelfi/).

- [Hawkes Regression](https://assets.researchsquare.com/files/rs-2146259/v1_covered.pdf?c=1666104480) - sum of splines for background.
