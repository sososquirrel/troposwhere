Salut Sophie,

I went through your notes. 

In short: 

I agree that the main result - the two distinct oscillation in the organized/non-organized regimes is independent of the number of clusters. S we could go with either. If it were up to me, I would lean strongly in favor. of the 5-mode version. 

The main advantages of the 5 modes are: 
- parcimony, as you said,
- the organized/unorganized cycles appear well separated. I also like that the second eigenvector (figure 4) captures the organized/unorganized convection nicely
- It would be easier to offer a physical interpretation of the 5 modes than of the 7. The 5-cluster seems to have a more direct interpretation here. From a practical perspective, your reader may stick with a discussion of 5 convetcive regimes, but is more likely to be tired with 7...

For the 7 modes, the primary benefit is that their distribution in the latent space appears more homogeneous (not sure if that's a good thing or a bad thing). Having two extra states may be useful, but I am struggling with their interpretation, although it may not be necessary for a ML paper. 

In more detail,

Question: When you mention PW, you mean the variance of PW, right? (i.e. your measure of the organization of the water vapor field)

Figure 1 and section 2: 
I agree - the shift from 5 to 7 mostly split two modes into 4, with some minor reshuffling. 

Figure 2:  For the 5-cluster case

Cluster 1: less low-level convection, but more ascent. This is consistent with what I've seen for convective cluster where the moist environment reduce the detrainment in the lower troposphere. 
Cluster 2: Very active, with large mass flux in the upper troposphere, consistent with present of meso-scale stratiform region

Cluster 3: Suppressed convection

Cluster 4: active convection, but with mass flux peaking lower than the organized convection (cluster 1-2)

Cluster 5: This looks like a suppressed convection, but with more active shallow convection 

(With 5 clusters, you could show the mean isentropic mass flux/streamfunction for each cluster.)

Figure 3. The transition matrix for the 5-mode case is quite nice, with th3 state 3 acting as the junction between two subregions (mode 1-2 vs 4-5). Modes 4-5 also seem to oscillate a bit between each other (the transitions 4->5 and 5->4 are fairly large), while 1 and 2 do not mix that much.   

Figure 4: I really like the structure of the second eigenvector, as it offers a nice interpretation of the 2nd eigenvalue as a separation between clusters 1-2 and clusters 4-5. (we could do something similar for the 7-clusters, but a few modes seems to be only weakly affected by the second eigenvector.

Let me know if you want to discuss this. I can do Monday mornign next week. I'll be at a conference from Tueday to Friday.

Cheers,

Olivier