add_executable(ActsUnitTestSeedfinderCudaMT SeedfinderCudaTestMT.cpp)
target_link_libraries(ActsUnitTestSeedfinderCudaMT PRIVATE ${unittest_extra_libraries} Boost::boost Boost::filesystem pthread)

# add_executable(ActsUnitTestSeedfinderCuda SeedfinderCudaTest.cpp)
# target_link_libraries(ActsUnitTestSeedfinderCuda PRIVATE ${unittest_extra_libraries} Boost::boost)

