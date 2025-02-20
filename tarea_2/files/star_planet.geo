// Parámetros
n = 5;                // Número de vértices de la estrella
r_max = 1;            // Radio máximo de la estrella
r_min = 0.4;          // Radio mínimo de la estrella
radius = 5.0;         // Radio del círculo exterior

// Ángulo base para los vértices de la estrella
theta = 2 * Pi / (2 * n);

// Primer punto en el eje y positivo (punta hacia arriba)
Point(1) = {r_max * Cos(Pi/2), r_max * Sin(Pi/2), 0, 0.1};

// Calcular los otros puntos de la estrella
For i In {1:n}
    // Puntos interiores de la estrella
    Point(2*i) = {r_min * Cos(Pi/2 + (2*i-1)*theta), r_min * Sin(Pi/2 + (2*i-1)*theta), 0, 0.1};
    
    // Puntos exteriores de la estrella
    Point(2*i+1) = {r_max * Cos(Pi/2 + 2*i*theta), r_max * Sin(Pi/2 + 2*i*theta), 0, 0.1};
EndFor

// Centro del círculo
Point(2*n+2) = {0, 0, 0, 1.0};

// Puntos del círculo exterior
For i In {1:n}
    Point(2*n+2+i) = {radius * Cos(Pi/2 + 2*i*theta), radius * Sin(Pi/2 + 2*i*theta), 0, 1};
EndFor

// Conectar los puntos para formar la estrella
For i In {1:n}
    Line(i) = {2*i-1, 2*i};          // Líneas internas de la estrella
    Line(n+i) = {2*i, 2*i+1};        // Líneas externas de la estrella
EndFor

// Cerrar el lazo de la estrella
Line(2*n+2) = {2*n, 1};             

// Bucle de líneas para la parte interior de la estrella
Line Loop(1) = {1, n+1, 2, n+2, 3, n+3, 4, n+4, 5, n+7}; // Interior de la estrella

// Crear la superficie dentro de la estrella
Plane Surface(1) = {1};        // Dentro de la estrella
Physical Surface(1) = {1};     // Superficie física dentro de la estrella

// Crear los círculos externos
For i In {1:n}
    If (i < n)
        Circle(2*n+2+i) = {2*n+2+i, 2*n+2, 2*n+2+1+i};  // Círculos conectados
    Else
        // El último elemento conecta con el primero
        Circle(2*n+2+i) = {2*n+2+1+i-1, 2*n+2, 2*n+2+1};  
    EndIf
EndFor

// Bucle de líneas para la parte exterior del círculo
Line Loop(2) = {2*n+2+1,2*n+2+2,2*n+2+3,2*n+2+4,2*n+2+5};  // Círculo exterior

// Crear la superficie fuera de la estrella
Plane Surface(2) = {2, 1};     // Fuera de la estrella
Physical Surface(2) = {2};     // Superficie física fuera de la estrella
