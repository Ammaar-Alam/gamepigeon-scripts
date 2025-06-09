#!/bin/bash

echo "Testing Capture Mode Solver with a standard starting position"
echo "Going first, standard board (4 stones in each pit)"
echo ""

echo "y
4 4 4 4 4 4 0 4 4 4 4 4 4 0" | java CaptureModeSolver

echo ""
echo "Press Enter to test with a mid-game position..."
read

echo "Testing with a mid-game position"
echo "Going second, custom board state"
echo ""

echo "n
3 0 2 5 1 0 8 2 4 0 3 1 5 12" | java CaptureModeSolver