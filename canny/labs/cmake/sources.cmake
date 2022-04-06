add_lab("CannyImage")
#add_lab_solution("CannyImage" ${PROJECT_SOURCE_DIR}/solution.cu)
set(sources ${PROJECT_SOURCE_DIR}/Otsus_Method_Sequential.cu ${PROJECT_SOURCE_DIR}/solution.cu)
cuda_add_executable(CannyImage_Solution ${sources})
target_link_libraries(CannyImage_Solution ${WBLIB} )

# Splitting out executables for my sanity

################################################################################
# Filters
################################################################################
set( target_filters filters )
set( sources_filters
  ${PROJECT_SOURCE_DIR}/filters.cu
  )
cuda_add_library( ${target_filters} ${sources_filters} )

################################################################################
# Otsu's
################################################################################
set( target_otsu otsu )
set( sources_otsu
  ${PROJECT_SOURCE_DIR}/Otsus_Method_Sequential.cu
  )
cuda_add_library( ${target_otsu} ${sources_otsu} )

################################################################################
# nonmaxsupp
################################################################################
set( sources_non_max_supp_serial
  )

################################################################################
# Build serial executables
################################################################################
set( target_canny_serial CannyImage_Serial )

set( canny_serial_libs
  ${target_filters}
  ${target_otsu}
  )

set( canny_serial_sources
  ${sources_non_max_supp_serial}
  ${PROJECT_SOURCE_DIR}/solution_serial.cpp
  ) 

add_executable( ${target_canny_serial} ${canny_serial_sources} )
target_link_libraries( ${target_canny_serial} ${WBLIB} ${canny_serial_libs} ) 

