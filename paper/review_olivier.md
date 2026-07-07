Pour le feedback,

 Section 1:  Intro.
Dans l'ensemble, c'est bon.
para 1. Je séparerais en deux paragraphes-  (1) convection as the main mechanism for upward energy and moisture transport, and (2) organization of convection. Organization broadly refers to structures within convective systems that persist beyond the convective life-time (a couple of hours). This allows you to view the slow evolution of the PDF of convection as a signature of organization
para 2: convective organization as a latent variable in convective closure (e.g. Khouider and Majda) (good)
Para 3: Limitation of the approach of para 2 - in my view, the primary issue here is that the model for convective organization is primarily empirical. 
para 4: statement of purpose for the paper - objective identification of the impact of convective organization on the upward energy transport
Para 5: I would expand, probably into two paragraphs - one broad review of isentropic analysis to characterize upward mass transport in convection, and a second as to why we are using it. (it allows for better capture of changes in the upward mass transport). 
para 6: I would probably expand this a little bit, depending on the target audience. Can you identify what aspect of the datasets are most challenging and motivate our approach (VAE with HMM)? 
para 7 and 8. I would rewrite this a bit. I like the overall ideas, but they need a bit more work. 
Section 2.2: Isentropic analysis
overall good.
I would explicitly separate figure 1 in 2 panels (a) and (b). You should discuss Figure 1b (I may have missed the discussion) in a way that fits the audience. How much thermodynamics/convection knowledge do we assume that the reader has? 
It may be useful to show snapshots of the isentropic analysis (and not simply the time average), so that the reader can get a better idea of what data the NN is actually processing. (e.g. you could make a big version of figure 1, showing 2 or 3 different snapshots of the cloud distribution and corresponding isentropic analysis)  
Section 3
I did not see any physicial interprestations of the cluster. Did you mean section 4.1?
VAE: I could use a bit more discussion of the VAE, namely, does the encoder predict both mu and sigma? (In my understanding of the VAE, the encoder predicts both, then you draw a random sample that you feed to the decoder, but I may get that wrong.)
HMM: the HMM is part of the NN. Can it be discovered directly from t a linear inverse model on the z_t? would that yield a different outcome?
On a practical side, having 7 clusters make a lot of the subsequent figures very busy. Can the analysis work with 4 or 5 clusters?
Section 4.1
Going back to the PCA of the latent space feels a bit awkward (I know we spent a lot of time on this, but as you've switiched to HMM, this breaks a bit the discussion of the paper. I also think you could get rid of the PCA. Here are my suggestion:
Redo figure 3, with the x and y axis being precipitation rate and organization index (PW - which should be explicitly defined and discussed as a measure of convective organization). Show the pdf of precip and OI for each cluster instead of PCA 1 and 2. 
Figure 4 becomes obsolete. The discussion of PCA could be moved to supplementary information.
I would like to see the isentropic mass flux  (or anomaly) for the cluster centroid, or at least a few of them (suggestion: clusters 2,3 5 and 6)
I don't like figure 5. By construction, you should have as much upward as downward mass flux - for panels a and b are redundant, and panel C should be 0 everywhere and is measuring round off error (or issues with the decoder that did not learn that the mass flux should cancel out...)
I would show instead (a) the upward mass transport, (2) the profile of MSE in the updraft and downdraft, and (3) the upward energy transport associated with each cluster.
Figure 6. Same issue as figure 5. I find the vertical integral of the mass flux less useful. I would rather show the net mass flux at selected lelevl, say 1km and 5km. 
Section 4.2. 
I would move the discussion of 'mean distribution (eq. 6) earlier. I would also rename the clusters based on their frequency.
Some time series of the cluster regime and transition would be quite useful for your reader.  I would select period during which l2 is either positive or negative. 
Can you estimate the projection of the data on lambda_2 (I don't think so, but there maybe a way to do it), or use a slow evolution of the statistics to capture this. 
I think one important thing to note is that the persistence associated with \ lambda1 outlasts the time associated with the probability transition. Am I correct? And if so, one can argue that lambda 2 can be viewed as a sign of a different mode of organization within the data. 
Section 4.3: I still need to go through this in detail. 
Can we make the case that the organization persists longer than predicted by the HMM's eigenvalues? 