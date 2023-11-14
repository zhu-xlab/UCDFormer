# UCDFormer
Change detection (CD) by comparing two bi-temporal images is a crucial task in remote sensing. With the advantages of requiring no cumbersome labeled change information, unsupervised CD has attracted extensive attention in the community. However,  existing unsupervised CD approaches rarely consider the seasonal  and style differences incurred by the illumination and atmospheric conditions in multi-temporal images. To this end, we propose a change detection with domain shift setting for remote sensing images. Furthermore, we present a novel  unsupervised CD method using a light-weight transformer, called UCDFormer. Specifically,  a transformer-driven image translation composed of a light-weight transformer and a domain-specific affinity weight is first proposed to mitigate domain shift between two images with real-time efficiency. After image translation, we can generate the difference map between the translated before-event image and the original after-event image. Then, a novel reliable pixel extraction module is proposed to select significantly changed/unchanged pixel positions by fusing the pseudo change maps of fuzzy c-means clustering and adaptive threshold. Finally, a binary change map is obtained based on these selected pixel pairs and a binary classifier.  Experimental results on different unsupervised CD tasks with seasonal  and style changes demonstrate the effectiveness of the proposed UCDFormer. For example, compared with several other related methods, UCDFormer improves performance on the Kappa coefficient by more than 12\%. In addition, UCDFormer achieves excellent performance for   earthquake-induced landslide detection when considering large-scale applications.
![Alt text](https://github.com/zhu-xlab/UCDFormer/blob/main/UCDFormer/Figures/fig0.jpg)
Fig. 1. Change detection with domain shift, which is used to reflect style differences and seasonal differences between multi-temporal images.
# Overview of UCDFormer
![Alt text](https://github.com/zhu-xlab/UCDFormer/blob/main/UCDFormer/Figures/fig1.jpg)
Fig. 2. Overview of UCDFormer.  The architecture of the model is divided into three parts, 	i.e., image translation, reliable pixel extraction, and change map extraction.  D, GN, CG, SBG represent downsampling, group normalization, channel group, and style bank generation, respectively.
# Datasets
We evaluate the proposed network on three datasets: data with seasonal differences, data with style differences, and earthquake-induced landslide detection with style differences.
# Pretrain Models
The Pretrain model of VGG is available at [Link](https://drive.google.com/file/d/1OIAOQv3zw5XHoPpcD2e7Nw39N5mYw74J/view?usp=drive_link)  
The Pretrain model of StyleFormer is available at [Link](https://drive.google.com/file/d/1E-0uBlHWBCwwNpydbKjDYsFnEZWtbubB/view?usp=drive_link)
# Run
After configuring the file path: run the code  
`python main_seasonal_change_from _summer_to_autumn.py` for Seasonal Change from Summer to Autumn  
`python main_seasonal_change_from_spring to_winter.py` for Seasonal Change from Spring to Winter  
`python main_style_changes.py` for Style Changes  
`python main_landslide.py` for Applications in Landslide Detection  
Other applications just change the corresponding data path.
# Citation
