function[A] = read_csr( file_ptrA, file_indA, file_valA, dir_in )
%
%[A] = read_csr( file_ptrA, file_indA, file_valA  [, dir_in] )
%
% 

dir = './';
if (nargin >= 4),
  dir = dir_in;
end;

has_slash = (dir(numel(dir)) == '/');
if (~has_slash),
  dir = strcat(dir,'/');
end;

ptrA = read_vec(strcat(dir,file_ptrA) );
indA = read_vec(strcat(dir,file_indA));
valA = read_vec(strcat(dir,file_valA) );

A = csr2sparse( ptrA, indA, valA );

