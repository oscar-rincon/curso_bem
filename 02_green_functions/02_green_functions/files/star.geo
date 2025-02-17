/*
Mesh to evaluate the potential due to a star shape
inside a circular domain.

Author: Adapted by ChatGPT
Date: February 2025
*/

num_points = 10;      // Number of points (5 outer, 5 inner)
radius_outer = 1.0;   // Outer radius of the star
radius_inner = 0.35;  // Inner radius of the star
lc = 100;             // Characteristic length for meshing

// Points for the Star (Aligned with y-axis)
For i In {0:num_points-1}
    theta = Pi/2 + i * Pi/5;  // Start at Pi/2 to align with y-axis
    If (i % 2 == 0)           // Outer points
        x = radius_outer * Cos(theta);
        y = radius_outer * Sin(theta);
    Else                      // Inner points
        x = radius_inner * Cos(theta);
        y = radius_inner * Sin(theta);
    EndIf
    Point(i+1) = {x, y, 0, lc};
EndFor

// Center Point for the circular domain
Point(11) = {0, 0, 0, lc};

// Define the Circular Boundary
radius_circle = 2.0;  // Radius of the circular domain
num_circle_points = 100; // Number of points for the circular boundary

// Points for the circle boundary
For i In {0:num_circle_points-1}
    theta = 2*Pi*i/num_circle_points;
    x = radius_circle * Cos(theta);
    y = radius_circle * Sin(theta);
    Point(12 + i) = {x, y, 0, lc};
EndFor

// Define Lines for the Star
For i In {1:num_points}
    If (i < num_points)
        Line(i) = {i, i+1};
    Else
        Line(i) = {i, 1};  // Close the loop of the star shape
    EndIf
EndFor

// Define Lines for the Circular Boundary (with explicit closure)
For i In {0:num_circle_points-1}
    Line(10 + i) = {12 + i, 12 + (i+1) % num_circle_points};
EndFor

 
// Define Surfaces
Line Loop(1) = {1:num_points};  // Star boundary
Plane Surface(1) = {1};         // Star surface

Line Loop(2) = {10:num_circle_points + 9};  // Circular boundary
Plane Surface(2) = {2};  // Circular domain surface

// Physical groups for post-processing
Physical Surface(1) = {1};   // Star
Physical Surface(2) = {2};   // Circular domain

// Mesh parameters
Transfinite Curve {1:num_points} = 20 Using Progression 1;  // Star boundary
Transfinite Curve {10:num_circle_points + 9} = 50 Using Progression 1; // Circular boundary
Transfinite Surface {1};  // Star region
Transfinite Surface {2};  // Circular domain region
