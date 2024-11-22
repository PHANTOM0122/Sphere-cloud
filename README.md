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


## :white_check_mark: Two public datasets!
- Indoor (**_7 Scenes_**): [Learning to navigate the energy landscape](https://graphics.stanford.edu/projects/reloc/) </br>
- Indoor (**_12 Scenes_**): [Learning to navigate the energy landscape](https://graphics.stanford.edu/projects/reloc/) </br>

## :running: How to run our code!
**Our code built upon the [repository of Paired-Point Lifting(PPL), CVPR2023](https://github.com/Fusroda-h/ppl/tree/main), accessed at June, 2023**. </br>
We borrowed most of the implementation of localization and inversion framework from PPL repository. </br>
Thanks to [Chunghwan Lee](https://github.com/Fusroda-h) for your contribution. </br>

- **Environment setting**

Make a new folder `/Myfolder`.
Make a docker container that fits your environment with a python version 3.9.
Mount the docker volume with the `-v /Myfolder/:/workspace/`.

:point_right: Clone the git `git clone https://github.com/PHANTOM0122/Ray-cloud`
Download `eigen-3.4.0.tar.gz` library from https://eigen.tuxfamily.org/index.php?title=Main_Page to run poselib.

```bash
cd Ray-cloud
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
```

:point_right: To properly build `poselib`, download the rest of the folders from the [PoseLib](https://github.com/vlarsson/PoseLib).
We only uploaded the customized code from PoseLib implementing P6L and P5+1R solver.

```bash
cd ..
git clone https://github.com/PoseLib/PoseLib.git
# Checkout to the version before refactoring "pybind"
cd PoseLib
git checkout ce7bf181731e4045f990c7e90e93716fe7465d56
# Overwrite customized local poselib to cloned poselib
# And move back to original directory
cd ../
cp -rf Ray-cloud/PoseLib/* PoseLib/
rm -r Ray-cloud/PoseLib
mv PoseLib Ray-cloud/PoseLib
```

:point_right: Since InvSfM code by Pittaluga et al. is written in tensorflow.v1, Chanhyuk Yun rewritten the whole code to pytorch for the ease of use ([invsfm_torch](https://github.com/ChanhyukYun/invSfM_torch)).
Download pretrained weights from [InvSfM](https://github.com/francescopittaluga/invsfm).
Position the `wts` folder to `utils/invsfm/wts`.
Then, our code will automatically change the weights to torch version and utilize it.

```bash
cd Ray-cloud
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
You can download example dataset on [Sample_data](https://1drv.ms/u/s!AlaAkmWU9TVG6yIqNBD0PlN43Ewe?e=2gIN1F).
Directories are organized like below.
```bash
├─Dataset_type (energy, cambridge)
│  └─Scene (apt1_living, kingscolledge)
│      ├─bundle_maponly
│      ├─images_maponly
│      ├─query
│      ├─sparse_gt
│      ├─sparse_maponly
│      └─sparse_queryadded
```
The construction of map and queries are explained in [here](documents/Lee_et_al_cvpr23_supplemat.pdf).

:point_right: To generate the each type of line cloud and to estimate pose & recover the point cloud from this

```
/usr/local/bin/python main.py
```

You can change your options with the parser in `main.py`.
Or else can manipulate the miute options with `static/variable.py`.

The results are stored in `output` folder.
In the folder, recovered point clouds, pose errors, and recovered image qualities are stored in `Dataset_name/Scene/L2Precon`,`Dataset_name/Scene/PoseAccuracy`,`Dataset_name/Scene/Quality` respectively.
The recovered images will be saved in `dataset/Dataset_name/Scene/invsfmIMG/`.

## Citation
```bibtex
@InProceedings{moon2024raycloud,
    author    = {Moon, Heejoon and Lee, Chunghwan and Hong, Je Hyeong},
    title     = {Efficient Privacy-Preserving Visual Localization Using 3D Ray Clouds},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {9773-9783}
}
```

## License
A patent application for the Raycloud algorithm and the relevant software has been submitted and is under review for registration(PCT).
Raycloud is licensed under the CC-BY-NC-SA-4.0 license limiting any commercial use.
[PoseLib](https://github.com/vlarsson/PoseLib) is licensed under the BSD 3-Clause license.

