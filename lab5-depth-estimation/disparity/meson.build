project('stereo', 'cpp',
  version: '0.1',
  default_options: [
    'cpp_std=c++20',
    'warning_level=3',
    'buildtype=debug',
    ])

opencv_dep = dependency('opencv4')

stereo_exe = executable('stereo',
  'main.cpp',
  'stereo_vision.hpp',
  dependencies: opencv_dep,
  )
