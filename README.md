# Sphere-cloud
## [BMVC 2024] Depth-Guided Privacy-Preserving Visual Localization Using 3D Sphere Cloud
**Authors:** Heejoon Moon, Jongwoo Lee, Jeonggon Kim, [Je Hyeong Hong](https://sites.google.com/view/hyvision)

**[[Paper]()][[Supplementary document]()]**

The emergence of deep neural networks capable of revealing high-fidelity scene details from sparse 3D point clouds has raised significant privacy concerns in visual localization involving private maps.
Lifting map points to randomly oriented 3D lines is a well-known approach for obstructing undesired recovery of the scene images, but these lines are vulnerable to a density-based attack that can recover the point cloud geometry by observing the neighborhood statistics of lines.
With the aim of nullifying this attack, we present a new privacy-preserving scene representation called \emph{sphere cloud}, which is constructed by lifting all points to 3D lines crossing the centroid of the map, resembling points on the unit sphere.
Since lines are most dense at the map centroid, the sphere cloud mislead the density-based attack algorithm to incorrectly yield points at the centroid, effectively neutralizing the attack. 
Nevertheless, this advantage comes at the cost of i) a new type of attack that may directly recover images from this cloud representation and ii) unresolved translation scale for camera pose estimation.
To address these issues, we introduce a simple yet effective cloud construction strategy to thwart new attack and 
 propose an efficient localization framework to guide the translation scale by utilizing absolute depth maps acquired from on-device time-of-flight (ToF) sensors.
Experimental results on public RGB-D datasets demonstrate sphere cloud achieves competitive privacy-preserving ability and localization runtime while not excessively compensating the pose estimation accuracy compared to other depth-guided localization methods.

https://github.com/user-attachments/assets/86c5dd48-4644-44f4-91d0-0dac732d01f3 

https://github.com/user-attachments/assets/cc71f34e-64dc-4b11-8534-02e178a6c5b2

*************************************
### :rocket: **News** 
:fire: [2024.11.23] We're released the intitial code!

##: Additional results beyond the paper

Below is the localization result following the prior works, where the camera positional error was reported in contrast to BMVC paper.
![image](https://github.com/user-attachments/assets/070d634d-4958-4f8b-a7a1-4cd0afc4bb7a)


## :white_check_mark: Two public datasets!
- Indoor [(**_7 Scenes_**)](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) </br>
- Indoor [(**_12 Scenes_**)](https://graphics.stanford.edu/projects/reloc/) </br> </br>
We used the evaluation benchmark from [Brachmann et al.](https://github.com/tsattler/visloc_pseudo_gt_limitations), using `dslam pGT`. 


## :running: How to build and run our code!
Clone the git <br>
```bash 
git clone https://github.com/PHANTOM0122/Sphere-cloud
```

Download `eigen-3.4.0.tar.gz` library from https://eigen.tuxfamily.org/index.php?title=Main_Page to run poselib.
```bash
cd Sphere-cloud
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
```
:point_right: Since InvSfM code by Pittaluga et al. is written in tensorflow.v1, Chanhyuk Yun rewritten the whole code to pytorch for the ease of use ([invsfm_torch](https://github.com/ChanhyukYun/invSfM_torch)).
Download pretrained weights from [InvSfM](https://github.com/francescopittaluga/invsfm).
Position the `wts` folder to `utils/invsfm/wts`.
Then, our code will automatically change the weights to torch version and utilize it.

```bash
bash start.sh
```

:warning: If you suffer from an initialization error with the message: `avx512fintrin.h:198:11: note: ‘__Y’ was declared here`.
Refer to this [ISSUE](https://github.com/pytorch/pytorch/issues/77939#issue-1242584624) and install with GCC-11
`apt-get install gcc-11 g++-11`
Edit the bash file `start.sh` so that Poselib is compiled with `gcc-11` $-$ substitute `cmake -S . -B _build/ -DPYTHON_PACKAGE=ON -DCMAKE_INSTALL_PREFIX=_install`
to `cmake -S . -B _build/ -DPYTHON_PACKAGE=ON -DCMAKE_INSTALL_PREFIX=_install -DCMAKE_C_COMPILER=/usr/bin/gcc-11 -DCMAKE_CXX_COMPILER=/usr/bin/g++-11`.

If you have other problems in building the packages.
Visit installation each page, s.t. [PoseLib](https://github.com/vlarsson/PoseLib), [Ceres-solver](http://ceres-solver.org/installation.html), [COLMAP](https://colmap.github.io/install.html).
Ubuntu and CUDA version errors might occur.

The codes `database.py` and `read_write_model.py` is from [COLMAP](https://github.com/colmap/colmap).
- **Run the main code (pose estimation, recovering point, restoring image at once)**

:white_check_mark:	
Directories of data are organized like below.
We offer sample dataset of 12scenes - apt1_kitchen. 
For inversion (pgt_triangulated), please download at [here](https://drive.google.com/drive/folders/1hZaNCcBGneO8yyu9oRTScDfUSePHxbZr?usp=sharing).
Test images are also available at [here](https://drive.google.com/file/d/1VkzdrxDFJVY5OoQNQ-veS22Mix-NtRke/view?usp=sharing).
```bash
├─data 
|  ├─dataset(12scenes, 7scenes)
│    └─Scene (apt1_living, chess)
│      ├─old_gt_(re)triangulated (From Brachman et al., ICCV 2021)
|          ├─ raw_depth.tar.gz: Calibrated Depth Images. Please unzip before localizaiton.
|          ├─ cameras.bin, images.bin, points3D.bin: COLMAP files
|          ├─ list_test.txt: Text of query images
|          ├─ points3D_with_fakeray_aug1_var0_10: Sphere cloud with 50% TP ratio
|          ├─ points3D_with_fakeray_aug2_var0_10: Sphere cloud with 33% TP ratio
|          ├─ points3D_with_fakeray_aug3_var0_10: Sphere cloud with 25% TP ratio
│      ├─pgt_triangulated (Custom dataset)
|          ├─ cameras.bin, cameras.txt, images.bin, images.txt, points3D.bin, points3D.txt, database.db
|          ├─ list_test.txt: Text of query images
|          ├─ points3D_with_fakeray_aug1_var0_10: Sphere cloud with 50% TP ratio
|          ├─ points3D_with_fakeray_aug2_var0_10: Sphere cloud with 33% TP ratio
|          ├─ points3D_with_fakeray_aug3_var0_10: Sphere cloud with 25% TP ratio
│      ├─test (RGB query images)
|         ├─rgb        
```

:point_right: Pose estimation 
```
bash localization_12scene.sh
bash localization_7scene.sh
```

:point_right: Inversion at test camera pose 
```
bash test_inversion_12scene.sh
bash test_inversion_7scene.sh
```

:point_right: Please refer to `static/variable.py` to manipulate options.

The results are stored in `output` folder.
In the folder, recovered point clouds, pose errors, and recovered image qualities are stored in `Dataset_name/Scene/L2Precon`,`Dataset_name/Scene/old_gt_(re)triangulated`,`Dataset_name/Scene/Quality` respectively.
The recovered images will be saved in `output/Dataset_name/Scene/invsfmIMG/`.

--> We have corrected the LPIPS measurement code after submitting the camera-ready paper. This correction does not affect the overall trends in our experimental results. We apologize for the mistake.

## Citation
```bibtex
@InProceedings{moon2024sphere,
    author    = {Moon, Heejoon and Lee, Jongwoo and Kim, Jeonggon and Hong, Je Hyeong},
    title     = {Depth-guided Privacy-Preserving Visual Localization Using 3D Sphere Clouds},
    booktitle = {Proceedings of the British Machine Vision Conference (BMVC)},
    year      = {2024},
}
```
## License
A patent application for the Spherecloud algorithm and the relevant software has been submitted and is under review for registration(PCT).
Spherecloud is licensed under the CC-BY-NC-SA-4.0 license limiting any commercial use.
[PoseLib](https://github.com/vlarsson/PoseLib) is licensed under the BSD 3-Clause license.

