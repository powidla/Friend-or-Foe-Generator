function tempmodel_community = createBaseTempmetmodel_community( ind_vec, filedr_metmodels )
%function takes vector of microbe indicies as an input and creates a tempmodel structure for
%Gurobi optimisation

%how many models:
nMs = length(ind_vec);

%find out how many reactions each microbe has and what the bmi is
nRxn_vec = zeros(1,nMs);
bmi_vec = zeros(1,nMs);
for ii = 1:nMs
    Mii = getHGTmodel_fromFile( ind_vec(ii), filedr_metmodels );
    nRxn_vec(ii) = length(Mii.lb);
    bmi_vec(ii) = Mii.bmi;
end
tnr = sum(nRxn_vec);
start_col_ind_vec = cumsum(nRxn_vec)+1;
start_col_ind_vec = [1,start_col_ind_vec(1:end-1)];
cum_bmi_vec = bmi_vec + start_col_ind_vec - 1;

num_ec = length(Mii.rhs_ext_lb);
num_nec = length(Mii.rhs_int_lb);

%set up empty temp model of the right size:
%stoichiometric matrix:
tempmodel_community.A = sparse( (2*num_ec + nMs*num_nec ), tnr );  %x2 rows for EC constraints, x1 for each internal metabolism and a constraint that everything grows above tolerance
tempmodel_community.lb = sparse(tnr,1);
tempmodel_community.ub = sparse(tnr,1);
tempmodel_community.rhs = zeros((2*num_ec + nMs*num_nec ),1);
tempmodel_community.sense = [repmat('>',num_ec,1);repmat('<',num_ec,1);repmat('=',nMs*num_nec,1)];
tempmodel_community.obj = zeros(tnr,1);

%run through each model and put in relevant info
for ii = 1:nMs

    Mii = getHGTmodel_fromFile( ind_vec(ii), filedr_metmodels );
    start_col_ind = start_col_ind_vec(ii);

    %shared environment:
    tempmodel_community.A(1:num_ec, start_col_ind:start_col_ind+nRxn_vec(ii)-1) = Mii.S_ext;
    tempmodel_community.A(num_ec+1:2*num_ec, start_col_ind:start_col_ind+nRxn_vec(ii)-1) = Mii.S_ext;
    %internal compartment:
    tempmodel_community.A( (2*num_ec + (ii-1)*num_nec + 1):(2*num_ec + (ii)*num_nec), start_col_ind:start_col_ind+nRxn_vec(ii)-1 ) = Mii.S_int;
    %flux constraints:
    tempmodel_community.lb(start_col_ind:start_col_ind+nRxn_vec(ii)-1) = Mii.lb;
    tempmodel_community.ub(start_col_ind:start_col_ind+nRxn_vec(ii)-1) = Mii.ub;
    %rhs:
    tempmodel_community.rhs(1:num_ec) = tempmodel_community.rhs(1:num_ec) + Mii.rhs_ext_lb;
    tempmodel_community.rhs(num_ec+1:2*num_ec) = tempmodel_community.rhs(num_ec+1:2*num_ec) + Mii.rhs_ext_ub;
    %objective include bmi:
    tempmodel_community.obj(cum_bmi_vec(ii)) = -1;
end

%lb and ub need to be 'full'
tempmodel_community.lb = full(tempmodel_community.lb);
tempmodel_community.ub = full(tempmodel_community.ub);

end
