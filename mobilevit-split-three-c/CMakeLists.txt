set(eapp_bin mobilevit-split-three)
set(eapp_src eapp/split1.c eapp/Network1.c eapp/reluLayer.c eapp/siLuLayer.c eapp/batchnormalLayer.c eapp/convLayer.c eapp/readdata1.c eapp/utils1.c eapp/layers_bn.c eapp/crypto.c eapp/aes.c)
set(host_bin mobilevit-split-three-runner)
set(host_src host/host.cpp host/aes.c)

set(str eapp/string.c)
set(sys eapp/syscall.c)
set(enc eapp/encret.s)

set(eapp_bin1 mobilevit-split-three-2)
set(eapp_src1 eapp/split2.c eapp/Network2.c eapp/fcLayer.c eapp/fcWeights.c eapp/reluLayer.c eapp/siLuLayer.c eapp/layers_ds.c eapp/batchnormalLayer2.c eapp/sigmoidLayer.c eapp/convLayer2.c eapp/utils2.c eapp/layers_bn2.c eapp/globalpoolLayer.c eapp/crypto.c eapp/aes.c)

set(eapp_bin2 mobilevit-split-three-3)
set(eapp_src2 eapp/split3.c eapp/Network3.c eapp/fcLayer.c eapp/fcWeights.c eapp/reluLayer.c eapp/siLuLayer.c eapp/layers_ds.c eapp/batchnormalLayer2.c eapp/sigmoidLayer.c eapp/convLayer2.c eapp/utils3.c eapp/layers_bn2.c eapp/globalpoolLayer.c eapp/crypto.c eapp/aes.c)

set(package_name "mobilevit-split-three.ke")
set(package_script "./mobilevit-split-three-runner mobilevit-split-three mobilevit-split-three-2 mobilevit-split-three-3 eyrie-rt --sm-bin fw_jump.bin")

if(RISCV32)
  set(eyrie_plugins "freemem rv32 io_syscall linux_syscall env_setup")
else()
  set(eyrie_plugins "freemem io_syscall linux_syscall env_setup")
endif()

# eapp

add_executable(${eapp_bin} ${eapp_src} ${str} ${sys} ${enc})
target_link_libraries(${eapp_bin} "-static")
target_include_directories(${eapp_bin}
  PUBLIC ${KEYSTONE_SDK_DIR}/include/app
  PUBLIC ${KEYSTONE_SDK_DIR}/include/edge)
  
add_executable(${eapp_bin1} ${eapp_src1} ${str} ${sys} ${enc})
target_link_libraries(${eapp_bin1} "-static")
target_include_directories(${eapp_bin1}
  PUBLIC ${KEYSTONE_SDK_DIR}/include/app
  PUBLIC ${KEYSTONE_SDK_DIR}/include/edge)
  
add_executable(${eapp_bin2} ${eapp_src2} ${str} ${sys} ${enc})
target_link_libraries(${eapp_bin2} "-static")
target_include_directories(${eapp_bin2}
  PUBLIC ${KEYSTONE_SDK_DIR}/include/app
  PUBLIC ${KEYSTONE_SDK_DIR}/include/edge)

# host

add_executable(${host_bin} ${host_src})
target_link_libraries(${host_bin}
  ${KEYSTONE_LIB_HOST} ${KEYSTONE_LIB_EDGE} ${KEYSTONE_LIB_VERIFIER} pthread)
set_target_properties(${host_bin}
  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO
  )
target_include_directories(${host_bin}
  PUBLIC ${KEYSTONE_SDK_DIR}/include/common
  PUBLIC ${KEYSTONE_SDK_DIR}/include/host
  PUBLIC ${KEYSTONE_SDK_DIR}/include/edge
  PUBLIC ${KEYSTONE_SDK_DIR}/include/verifier)

# add target for Eyrie runtime (see keystone.cmake)

set(eyrie_files_to_copy .options_log eyrie-rt)
add_eyrie_runtime(${eapp_bin}-eyrie
  ${eyrie_plugins}
  ${eyrie_files_to_copy})

# add target for packaging (see keystone.cmake)

add_keystone_package(${eapp_bin}-package
  ${package_name}
  ${package_script}
  ${eyrie_files_to_copy} ${eapp_bin} ${eapp_bin1} ${eapp_bin2} ${host_bin} ${fw_bin} ${image})

add_dependencies(${eapp_bin}-package ${eapp_bin}-eyrie)

# add package to the top-level target
add_dependencies(examples ${eapp_bin}-package)
