%%set up

collection = 'CarveMe';
colAbr = 'CM';
tot_nMs = 5587;
nECtot = 499;

%met model folder:
filedr_metmodels =  ['path should be here...' collection 'collection should be here...'];
% %path for functions:
% pathWcommonfuns = 'path should be here...';
% addpath(pathWcommonfuns)

%microbe indices:
%first N species
N = 50;
ind_vec = 1:N;
% %could instead have N random microbes, e.g.
% ind_vec = datasample(1:tot_nMs, N, 'Replace',false);

%build tempmodel for community:
tempmodel_community = createBaseTempmetmodel_community( ind_vec, filedr_metmodels );

%%this is where you can change the environment:
%update the environment (divide by no. species)
env_rhslb = tempmodel_community.rhs(1:nECtot)/length(ind_vec);
tempmodel_community.rhs(1:nECtot) = env_rhslb;

tic

%grow the community:
params = struct();
params.OutputFlag = 0;
%optimise:
result_community = gurobi(tempmodel_community,params);
%individual growth rates:
bmi_id = tempmodel_community.obj;
indiv_grs = result_community.x(bmi_id==-1);

toc

disp(['Growing without constraints found that ', num2str(sum(indiv_grs>0.1)), ' microbes could grow at a rate of above 0.1 in a community of ', num2str(length(ind_vec)), ' species'])

%implement that everyone has to grow above g_tol:
%growth tolerance:
g_tol = 0.1;
tempmodel_community2 = createBaseTempmetmodel_community_everyoneGrows( ind_vec, filedr_metmodels, g_tol );
%update the environment (divide by no. species)
tempmodel_community2.rhs(1:nECtot) = env_rhslb;
result_community2 = gurobi(tempmodel_community2,params);
%individual growth rates:
bmi_id_2 = tempmodel_community2.obj;
indiv_grs2 = result_community.x(bmi_id_2==-1);

if abs(sum(indiv_grs) - sum(indiv_grs)) > 1*10^-6
    disp('check result when ensuring that all microbes can grow')
end

%compare for an individuals:
indiv_grs_solo = zeros(length(ind_vec),1);

for ii = 1:length(ind_vec)
    M1 = getHGTmodel_fromFile( ind_vec(ii), filedr_metmodels );
    num_nec = length(M1.rhs_int_lb);
    tempmodelMi = createTempmetmodelMi_noenvrhs( M1, nECtot, num_nec );
    env_rhsub = M1.rhs_ext_ub;
    tempmodelMi.rhs(1+nECtot:2*nECtot) = env_rhsub; %could instead set this to be the same as the community value
    tempmodelMi.rhs(1:nECtot) = env_rhslb;          %make sure this is the same as for the community for a fair comparison
    result_indiv = gurobi(tempmodelMi,params);

    indiv_grs_solo(ii) = abs(result_indiv.objval);
end

disp([num2str(sum(indiv_grs>indiv_grs_solo)), ' microbes can grow faster in the community than alone'])
