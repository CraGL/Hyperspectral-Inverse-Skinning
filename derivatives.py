# A good primer on basic matrix calculus: https://atmos.washington.edu/~dennis/MatrixCalculus.pdf
# The Matrix Reference Manual: http://www.ee.ic.ac.uk/hp/staff/dmb/matrix/intro.html#Intro
# Trying to understand the derivative of the inverse: https://math.stackexchange.com/questions/1471825/derivative-of-the-inverse-of-a-matrix
# Derivative of the pseudoinverse:
    https://math.stackexchange.com/questions/2179160/derivative-of-pseudoinverse-with-respect-to-original-matrix
    https://mathoverflow.net/questions/25778/analytical-formula-for-numerical-derivative-of-the-matrix-pseudo-inverse
    https://mathoverflow.net/questions/264130/derivative-of-pseudoinverse-with-respect-to-original-matrix/264426
    https://math.stackexchange.com/questions/1689434/derivative-of-the-frobenius-norm-of-a-pseudoinverse-matrix
# Math Overflow user john316 does derivatives with Frobenius norms ( https://math.stackexchange.com/users/262158/john316 ):
    https://math.stackexchange.com/questions/1689434/derivative-of-the-frobenius-norm-of-a-pseudoinverse-matrix
    https://math.stackexchange.com/questions/1405922/what-is-the-gradient-of-f-s-abat-2/1406290#1406290
    https://math.stackexchange.com/questions/946911/minimize-the-frobenius-norm-of-the-difference-of-two-matrices-with-respect-to-ma/1474048#1474048
# Math Overflow user greg also does derivatives with Frobenius norms ( https://math.stackexchange.com/users/357854/greg ):
    https://math.stackexchange.com/questions/2444284/matrix-derivative-of-frobenius-norm-with-hadamard-product-inside
    https://math.stackexchange.com/questions/1890313/derivative-wrt-to-kronecker-product/1890653#1890653
# Some matrix calculus:
    Practical Guide to Matrix Calculus for Deep Learning (Andrew Delong)
    http://www.psi.toronto.edu/~andrew/papers/matrix_calculus_for_learning.pdf

# Properties (: is Frobenius inner product, ∘ and ⊙ are element-wise Hadamard product, * is matrix multiplication):
    d(X:X)=dX:X+X:dX=2X:dX
    A:B=B:A
    A:(B+C)=A:B + A:C
    A∘B=B∘A
    A:B∘C=A∘B:C
    A:(B*C) = (B'*A):C = (A*C'):B
    d(X:Y) = (dX):Y + X:(dY)
    d(X⊙Y) = (dX)⊙Y + X⊙(dY)
    d(X*Y) = (dX)*Y + X*(dY)
    d(X') = (dX)'
    dZ/dX = dZ/dY * dY/dX
    d(inv(X)) = -inv(X)*dX*inv(X)

E = norm2(v*(p+B*-inv(B'*v'*v*B)*B'*v'*(v*p-w))-w)^2
  = norm2((v*p-w)-v*B*inv(B'*v'*v*B)*B'*v'*(v*p-w))^2

S = B'*v'*v*B
    S = S'
    inv(S) = inv(S)'
u = v*p-w
R = v*B*inv(S)
Q = R*B'*v' = v*B*inv(S)*B'*v'
M = (I-Q)*u = u - Q*u

E  = M : M
dE = 2*M : dM
   = 2*M : d[ (v*p-w) - v*B*inv(S)*B'*v'*(v*p-w) ]
   = 2*M : (v*dp - Q*v*dp) + 2*M : (-v*dB*R'*u) + 2*M : (-v*B*[d inv(S)]*B'*v'*u) + 2*M : (-R*dB'*v'*u)
   = 2*M : (v*dp - Q*v*dp) + 2*M : (-v*dB*R'*u) + 2*M : (R*[dS]*R'*u) + 2*M : (-R*dB'*v'*u)

dS = d( B'*v'*v*B ) = dB'*v'*v*B + B'*v'*v*dB

dE = 2*M : (v*dp - Q*v*dp) + 2*M : (-v*dB*R'*u) + 2*M : (R*[dB'*v'*v*B + B'*v'*v*dB]*R'*u) + 2*M : (-R*dB'*v'*u)
   = 2*M : (v - Q*v)*dp    + 2*M : (-v*dB*R'*u) + 2*M : (R*[dB'*v'*v*B + B'*v'*v*dB]*R'*u) + 2*M : (-R*dB'*v'*u)
   = 2*(v - Q*v)'*M : dp   + 2*M : (-v*dB*R'*u) + 2*M : (R*[dB'*v'*v*B + B'*v'*v*dB]*R'*u) + 2*M : (-R*dB'*v'*u)
   = 2*(v - Q*v)'*M : dp   - 2*M : ( v*dB*R'*u) + 2*M : (R*[dB'*v'*v*B + B'*v'*v*dB]*R'*u) - 2*M : ( R*dB'*v'*u)
   = 2*(v - Q*v)'*M : dp   - 2*v'*M*(R'*u)' :dB + 2*R'*M*(R'*u)' : (dB'*v'*v*B+B'*v'*v*dB)  - 2*R'*M*(v'*u)' : dB'
   = 2*(v - Q*v)'*M : dp   - 2*v'*M*(R'*u)' :dB + 2*R'*M*(R'*u)' : (dB'*v'*v*B+B'*v'*v*dB)  - 2*(v'*u)*M'*R : dB
   = 2*(v - Q*v)'*M : dp   - 2*v'*M*(R'*u)' :dB + 2*R'*M*(R'*u)' : dB'*v'*v*B + 2*R'*M*(R'*u)' : B'*v'*v*dB - 2*(v'*u)*M'*R : dB
   = 2*(v - Q*v)'*M : dp   - 2*v'*M*(R'*u)' :dB + 2*R'*M*(R'*u)'*(v'*v*B)' : dB' + 2*(v'*v*B)*R'*M*(R'*u)' : dB - 2*(v'*u)*M'*R : dB
   = 2*(v - Q*v)'*M : dp   - 2*v'*M*(R'*u)' :dB + 2*(v'*v*B)*R'*u*M'*R : dB + 2*(v'*v*B)*R'*M*(R'*u)' : dB - 2*(v'*u)*M'*R : dB

dE/dp = 2*(v - Q*v)'*M

dE/dB = - 2*v'*M*(R'*u)'   + 2*(v'*v*B)*R'*u*M'*R + 2*(v'*v*B)*R'*M*(R'*u)' - 2*(v'*u)*M'*R
      = -2*( v'*M*(R'*u)' - (v'*v*B)*R'*u*M'*R - (v'*v*B)*R'*M*(R'*u)' + (v'*u)*M'*R )
      = -2*v'*( M*(R'*u)' - v*B*R'*u*M'*R - v*B*R'*M*(R'*u)' + u*M'*R )
      = -2*v'*( M*(R'*u)' - v*B*R'*u*M'*R - Q*M*(R'*u)' + u*M'*R )
      = -2*v'*( M*u'*R - Q*u*M'*R - Q*M*u'*R + u*M'*R )
