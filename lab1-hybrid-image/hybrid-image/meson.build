project('hybrid', 'cpp',
  version: '0.1',
  default_options: [
    'cpp_std=c++20',
    'warning_level=2',
    'buildtype=release',
  ],
)

opencv_dep = dependency('opencv4')
hybrid_exe = executable('hybrid',
  'hybrid-image.cpp',
  dependencies: opencv_dep)
