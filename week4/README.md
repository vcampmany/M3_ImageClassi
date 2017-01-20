# Week 3

First of all install yael (Linux):
 - Download https://gforge.inria.fr/frs/download.php/file/34217/yael_v438.tar.gz and extract
 - inside the folder run `./configure.sh` and `make`
 - now update PYTHONPATH by `export PYTHONPATH=$PYTHONPATH:/path/of/download/yael_v438`

 If there is an error in yael library when using pca, do the following:
 Change the following on the file `gmm.c` of yael library:
 Substitute

    float is = 1.0 / s;
    for(l = 0; l < k; l++) 
    	P(l, i) *= is;

by

    if(s != 0) {
    	float is = 1.0 / s;
   		for(l = 0; l < k; l++)
    		P(l, i) *= is;
    }

 Source: https://gforge.inria.fr/forum/forum.php?thread_id=33378&forum_id=6337&group_id=2151

To call with pyramid add:
`-pyramid 2 2` To create a spatial pyramid of 2 by 2