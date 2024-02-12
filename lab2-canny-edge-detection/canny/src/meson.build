canny_dir = 'canny/'
util_dir = 'util/'

canny_exe = executable('canny',
  'main.cpp',
  canny_dir + 'canny.cpp',
  util_dir + 'util.cpp',
  include_directories: [canny_dir, util_dir],
  dependencies: opencv_dep)
