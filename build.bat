@echo off
IF NOT EXIST "bin" mkdir bin

gcc -Wall -Wextra -O3 -Iinclude ^
    src/neural_net.c ^
    src/data_loader.c ^
    src/main.c ^
    -o bin/mnist_nn.exe ^
    -lm -static-libgcc -static-libstdc++

IF %ERRORLEVEL% EQU 0 (
    echo Build successful! The executable is in the bin directory.
) ELSE (
    echo Build failed!
)
