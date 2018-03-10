echo "generate test points"
g++ --std=c++17 points_generator_test.cpp
./a.out
echo "generate train points"
g++ --std=c++17 points_generator_train.cpp
./a.out
rm a.out
