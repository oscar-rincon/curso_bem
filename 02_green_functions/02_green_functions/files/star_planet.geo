// Parameters
n = 5;
r_max = 1;
r_min = 0.4;
radius = 5.0;

// Base angle
theta = 2 * Pi / (2 * n);

// First point at the positive y-axis (tip up)
Point(1) = {r_max * Cos(Pi/2), r_max * Sin(Pi/2), 0, 0.1};

// Calculate the other points
For i In {1:n}
    Point(2*i) = {r_min * Cos(Pi/2 + (2*i-1)*theta), r_min * Sin(Pi/2 + (2*i-1)*theta), 0, 0.1};
    Point(2*i+1) = {r_max * Cos(Pi/2 + 2*i*theta), r_max * Sin(Pi/2 + 2*i*theta), 0, 0.1};
EndFor


// Rest of the outer circle points
//For i In {1:n}
//    Point(2*n+2+i) = {radius * Cos(Pi/2 + 2*i*theta), radius * Sin(Pi/2 + 2*i*theta), 0, 1};
//EndFor

// Connect points to form the star
For i In {1:n}
    Line(i) = {2*i-1, 2*i};           // Inner lines of the star
    Line(n+i) = {2*i, 2*i+1};          // Outer lines of the star
EndFor
Line(2*n+2) = {2*n, 1};              // Close the star loop


Line Loop(1) = {1, n+1, 2, n+2, 3, n+3, 4, n+4, 5, n+7}; // Star interior
 

// Surfaces
Plane Surface(1) = {1};        // Inside the star

// Physical groups
Physical Surface(1) = {1};     // Inside the star


// Center of the circle
Point(2*n+2) = {0, 0, 0, 1.0};


// Rest of the outer circle points
For i In {1:n}
    Point(2*n+2+i) = {radius * Cos(Pi/2 + 2*i*theta), radius * Sin(Pi/2 + 2*i*theta), 0, 1};
EndFor

// Create the circles iteratively
For i In {1:n}
    If (i < n)
        Circle(2*n+2+i) = {2*n+2+i, 2*n+2, 2*n+2+1+i};
    Else
        // Last element wraps around to the first point
        Circle(2*n+2+i) = {2*n+2+1+i-1, 2*n+2, 2*n+2+1};
    EndIf
EndFor

Line Loop(2) = {2*n+2+1,2*n+2+2,2*n+2+3,2*n+2+4,2*n+2+5};  // Circle

// Physical groups
 
 
Plane Surface(2) = {2, 1};     // Outside the star
Physical Surface(2) = {2};     // Outside the star


 