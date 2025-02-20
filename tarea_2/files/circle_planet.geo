// Parámetros
n = 20;                 // Número de puntos para el círculo interno y externo
radius_inner = 1.0;     // Radio del círculo interno
radius_outer = 5.0;     // Radio del círculo exterior

// Ángulo base para distribuir los puntos en los círculos
theta = 2 * Pi / n;

// ---------------------------
// Definición del círculo interno
// ---------------------------

// Centro del círculo interno
Point(1) = {0, 0, 0, 1.0};

// Generar los puntos del círculo interno
For i In {1:n}
    Point(1+i) = {radius_inner * Cos((i-1)*theta), radius_inner * Sin((i-1)*theta), 0, 0.2};
EndFor

// Crear los arcos que forman el círculo interno
For i In {1:n-1}
    Circle(i) = {1+i, 1, 2+i};  // Centro en el punto 1
EndFor
Circle(n) = {n+1, 1, 2};        // Último arco, cerrando el círculo

// Definir el loop y la superficie del círculo interno
Line Loop(1) = {1:n};
Plane Surface(1) = {1};
Physical Surface(1) = {1};  // Superficie física del círculo interno

// ---------------------------
// Definición del círculo externo
// ---------------------------

// Centro del círculo exterior
Point(n+2) = {0, 0, 0, 1.0};

// Generar los puntos del círculo exterior
For i In {1:n}
    Point(n+2+i) = {radius_outer * Cos((i-1)*theta), radius_outer * Sin((i-1)*theta), 0, 1.0};
EndFor

// Crear los arcos que forman el círculo exterior
For i In {1:n-1}
    Circle(n+2+i) = {n+2+i, n+2, n+2+i+1};  // Centro en el punto n+2
EndFor
Circle(n+2+n) = {n+2+n, n+2, n+3};          // Último arco, cerrando el círculo

// Definir el loop y la superficie del círculo exterior
Line Loop(2) = {n+3:2*n+2};
Plane Surface(2) = {2, 1};  // Superficie exterior, restando la interna
Physical Surface(2) = {2};  // Superficie física del círculo exterior
