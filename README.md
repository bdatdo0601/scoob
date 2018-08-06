# Scoob

Parallel Programming project to cartoonize images

## Pre-requirement

Basically ensure you have CMake, C++ Compiler, and OpenCV

### Mac

- [XCode](https://developer.apple.com/xcode/)
  - Make sure to accept the lisence by typing `sudo xcodebuild -license` in terminal and follow instruction
- Apple Command Line Tools - `sudo xcode-select --install` in terminal
- [Homebrew](https://brew.sh/) - Mac OSX Package Manager
  - Once installed, update with `brew update`
  - Go to your shell profile file (default is bash `nano ~/.bash-profile`) and included this line at the end of the file then run `source <your-profile-file>` on terminal

```bash
# Homebrew
export PATH=/usr/local/bin:$PATH
```

- Python/Python3 - This is not required but might as well for future development
  - `brew install python python3`
  - `brew linkapps python` and `brew linkapps python3`
  - Verify python is installed by typing `which python` and `which python3`
- OpenCV - `brew install opencv3`

### Windows

Haven't tested but there is this post that detail on how to install OpenCV & CMake
[https://www.learnopencv.com/install-opencv3-on-windows/](https://www.learnopencv.com/install-opencv3-on-windows/)

## Installation

1. Open your Terminal or Powershell, locate to the repository
2. Create a build folder (`mkdir build`)
3. Change directory to build folder (`cd build/`)
4. Generate makefiles with CMake (`cmake -D CMAKE_CXX_COMPILER=g++-8 -D CMAKE_C_COMPILER=gcc-8 ..`). This should take a while
5. Once done, a Makefile will be generated. Compile the project (`make -j4`). `-j4` tell your machine to compile with all 4 cores to increase speed. This should take a while
6. Once completed, you will be left with a single executable `Scoobs`, which is our project.

## Usage

We use Dlib library to do detection on face and edges in our images. The way it works is that it uses a trained model to detect stuff. With that being said, we would need a pre-trained model for that. You can download it [here](https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat) (the existing pre-trained model on this repository might be corrupted so please get that one and override it)

For now, `Scoobs` will take in 2 to many arguments, the pre-trained model file (`shape_predictor_68_face_landmarks.dat`) and as many images as you want.

```bash
./Scoobs <path_to_pretrained_model> <path_to_image_1> <path_to_image_2> ... <path_to_image_n>
```