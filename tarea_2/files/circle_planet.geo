// Parameters
n = 20;                 // Número de puntos para el círculo interno
radius_inner = 1.0;     // Radio del círculo interno
radius_outer = 5.0;     // Radio del círculo exterior

// Ángulo base para el círculo interno
theta = 2 * Pi / n;

// Centro del círculo
Point(1) = {0, 0, 0, 1.0};

// Puntos del círculo interior
For i In {1:n}
    Point(1+i) = {radius_inner * Cos((i-1)*theta), radius_inner * Sin((i-1)*theta), 0, 0.2};
EndFor

// Crear el círculo interior usando arcos
For i In {1:n-1}
    Circle(i) = {1+i, 1, 2+i};   // Centro en 1, arco entre puntos consecutivos
EndFor
Circle(n) = {n+1, 1, 2};         // Último arco, cerrando el círculo

// Definir el Loop del círculo interior
Line Loop(1) = {1:n};
Plane Surface(1) = {1};
Physical Surface(1) = {1};     // Interior del círculo


// Centro del círculo exterior
Point(n+2) = {0, 0, 0, 1.0};

// Puntos del círculo exterior
For i In {1:n}
    Point(n+2+i) = {radius_outer * Cos((i-1)*theta), radius_outer * Sin((i-1)*theta), 0, 1.0};
EndFor

// Crear el círculo exterior usando arcos
For i In {1:n-1}
    Circle(n+2+i) = {n+2+i, n+2, n+2+i+1};   // Centro en n+2
EndFor
Circle(n+2+n) = {n+2+n, n+2, n+3};           // Último arco, cerrando el círculo

// Definir el Loop del círculo exterior
Line Loop(2) = {n+3:2*n+2};
Plane Surface(2) = {2, 1};
Physical Surface(2) = {2};     // Exterior del círculo
