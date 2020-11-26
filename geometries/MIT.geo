// Variables
E = 1.0;      // Element size
H = 0.0;      // Height (Initialized)
j = 0;        // Switch

// Bottom points
BP1 = newp; Point(BP1) = {0,   H, 0, E};
BP2 = newp; Point(BP2) = {1/6, H, 0, E};
BP3 = newp; Point(BP3) = {1/2, H, 0, E};
BP4 = newp; Point(BP4) = {5/6, H, 0, E};
BP5 = newp; Point(BP5) = {1,   H, 0, E};

// Bottom lines
BL1 = newl; Line(BL1) = {BP1, BP2}; Transfinite Line {BL1} = 0 Using Progression 1;
BL2 = newl; Line(BL2) = {BP2, BP3}; Transfinite Line {BL2} = 0 Using Progression 1;
BL3 = newl; Line(BL3) = {BP3, BP4}; Transfinite Line {BL3} = 0 Using Progression 1;
BL4 = newl; Line(BL4) = {BP4, BP5}; Transfinite Line {BL4} = 0 Using Progression 1;

// Bottom boundary
Physical Line(3) = {BL1, BL2, BL3, BL4};

For n In {0:21}
  // Increment height
  If (n == 0 || n == 21)
    H += 1/6;
  ElseIf (n == 1 || n == 2 || n == 19 || n == 20 )
    H += 2/9;
  ElseIf (n > 2 || n < 19)
    H += 61/144;
  EndIf

  If ( n > 0)
    j = 1;
  EndIf

  // Upper row points
  P1 = newp; Point(P1) = {0,   H, 0, E};
  P2 = newp; Point(P2) = {1/6, H, 0, E};
  P3 = newp; Point(P3) = {1/2, H, 0, E};
  P4 = newp; Point(P4) = {5/6, H, 0, E};
  P5 = newp; Point(P5) = {1,   H, 0, E};

  // Upper row lines
  L1 = newl; Line(L1) = {P1, P2}; Transfinite Line {L1} = 0 Using Progression 1;
  L2 = newl; Line(L2) = {P2, P3}; Transfinite Line {L2} = 0 Using Progression 1;
  L3 = newl; Line(L3) = {P3, P4}; Transfinite Line {L3} = 0 Using Progression 1;
  L4 = newl; Line(L4) = {P4, P5}; Transfinite Line {L4} = 0 Using Progression 1;

  // Vertical lines
  L5 = newl; Line(L5) = {P1 - 5, P1}; Transfinite Line {L5} = 0 Using Progression 1;
  L6 = newl; Line(L6) = {P2 - 5, P2}; Transfinite Line {L6} = 0 Using Progression 1;
  L7 = newl; Line(L7) = {P3 - 5, P3}; Transfinite Line {L7} = 0 Using Progression 1;
  L8 = newl; Line(L8) = {P4 - 5, P4}; Transfinite Line {L8} = 0 Using Progression 1;
  L9 = newl; Line(L9) = {P5 - 5, P5}; Transfinite Line {L9} = 0 Using Progression 1;

  // Perimeter of each surface
  LL1 = newll; Line Loop(LL1) = {-L5, L1 - 4 - 13*j, L6, -L1};
  LL2 = newll; Line Loop(LL2) = {-L6, L2 - 4 - 13*j, L7, -L2};
  LL3 = newll; Line Loop(LL3) = {-L7, L3 - 4 - 13*j, L8, -L3};
  LL4 = newll; Line Loop(LL4) = {-L8, L4 - 4 - 13*j, L9, -L4};
  
  // Surfaces
  S1 = news; Plane Surface(S1) = {LL1}; 
  S2 = news; Plane Surface(S2) = {LL2};
  S3 = news; Plane Surface(S3) = {LL3};
  S4 = news; Plane Surface(S4) = {LL4};
EndFor

// Top boundary
Physical Line(4) = {L1, L2, L3, L4};

// Left boundary
Physical Line(1) = {366, 349, 332, 315, 298, 281, 60, 128, 111, 94, 77, 43, 26, 9, 145, 162, 179, 196, 213, 230, 247, 264};

// Right boundary
Physical Line(2) = {285, 200, 217, 234, 251, 268, 183, 302, 319, 336, 353, 370, 13, 30, 47, 64, 81, 98, 115, 132, 149, 166};

// Domain
Physical Surface(1) = {38, 72, 71, 70, 69, 55, 54, 53, 52, 37, 36, 35, 21, 20, 19, 18, 86, 274, 308, 307, 293, 292, 291, 290, 276, 275, 309, 273, 259, 258, 257, 256, 242, 241, 240, 344, 378, 377, 376, 375, 361, 360, 359, 358, 239, 343, 342, 341, 327, 326, 325, 324, 310, 121, 155, 154, 140, 139, 138, 137, 123, 122, 156, 120, 106, 105, 104, 103, 89, 88, 87, 191, 225, 224, 223, 222, 208, 207, 206, 205, 190, 189, 188, 174, 173, 172, 171, 157};

Recombine Surface {38, 72, 71, 70, 69, 55, 54, 53, 52, 37, 36, 35, 21, 20, 19, 18, 86, 274, 308, 307, 293, 292, 291, 290, 276, 275, 309, 273, 259, 258, 257, 256, 242, 241, 240, 344, 378, 377, 376, 375, 361, 360, 359, 358, 239, 343, 342, 341, 327, 326, 325, 324, 310, 121, 155, 154, 140, 139, 138, 137, 123, 122, 156, 120, 106, 105, 104, 103, 89, 88, 87, 191, 225, 224, 223, 222, 208, 207, 206, 205, 190, 189, 188, 174, 173, 172, 171, 157};
RecombineMesh;
Coherence Mesh;
//+
//+
