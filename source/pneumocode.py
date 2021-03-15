"""

pneumocode - An all-in-one inference and stochastic simulation class for pneumococcus transmission models in python.

This is the source code for the class in which you can find comments on how pneumocode works internally. Please 
also feel free to consult the Jupyter Notebooks for a more user-friendly experience.


"""

# Necessary imports - these python modules must be installed to use the pneumocode class
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as spec


# Initialise the 'pneumocode' method class
class pneumocode:


    def __init__(self,model_mode,num_of_serotypes,suppress_terminal_output=False):
        """

        pneumocode - An all-in-one inference and stochastic simulation class for pneumococcus transmission models in python.

        Args:

        [Mandatory] -       model_mode         - The mode for the modelling approach. Choose from: 
                                                 'fixed': the transmission rates are independent of the population state
                                                 'vary': the transmission rates depend on the population state

        [Mandatory] -    num_of_serotypes      - The effective number of serotypes (separate disease states) that each
                                                 person can exist in. This does not include the disease-free state.

        [Optional]  - suppress_terminal_output - If set to True then no terminal output will be displayed for pneumocode
                                                 throughout its use. The default setting is False.
     
        """

        # Set initialisation variables of the class
        self.mode = model_mode
        self.nsero = num_of_serotypes
        self.npeop = 0
        self.supp_out = suppress_terminal_output

        # Directory names can be changed here if necessary
        self.chains_path = 'chains/' 
        self.output_path = 'data/'
        self.plots_path = 'plots/'
        self.source_path = 'source/'

        # If the transmission rates are independent of the population state then setup the appropriate dictionaries
        if self.mode == 'fixed':

            # Define an empty dictionary for the population members which includes their parameters and loop 
            # over the number of serotypes to create all of the dictionary columns
            self.pop = {'sig' : [], 'eps' : [], 'mumax' : [], 'Curr' : []} 
            self.ode_pop = {'sig' : [], 'eps' : [], 'mumax' : [], 'Curr' : [], 'N' : []}
            for ns in range(1,self.nsero+1):
                self.pop['npast_' + str(ns)] = []  # The number of past colonisations by this serotype
                self.pop['Lam_' + str(ns)] = []    # The colonisation rate of this serotype
                self.pop['mu_' + str(ns)] = []     # The recovery rate from this serotype
                self.pop['f_' + str(ns)] = []      # The competitiveness factor of this serotype
                self.pop['vef_' + str(ns)] = []    # The vaccine efficacy against this serotype
                self.pop['vt_' + str(ns)] = []     # The time of vaccination against this serotype
                self.ode_pop['npast_' + str(ns)] = []  # The number of past colonisations by this serotype
                self.ode_pop['Lam_' + str(ns)] = []    # The colonisation rate of this serotype
                self.ode_pop['mu_' + str(ns)] = []     # The recovery rate from this serotype
                self.ode_pop['f_' + str(ns)] = []      # The competitiveness factor of this serotype
                self.ode_pop['vef_' + str(ns)] = []    # The vaccine efficacy against this serotype
                self.ode_pop['vt_' + str(ns)] = []     # The time of vaccination against this serotype

        # Methods of the pneumocode class
        self.create_people
        self.fit_data
        self.run_ode
        self.run_sim


    def create_people(self,num_of_people,parameter_dic):
        """

        Method to add people (or a person) to the population.

        Args:

        [Mandatory] - num_of_people  - The number of people in this added group to the population.

        [Mandatory] - parameter_dic  - A dictionary of parameters and parameter arrays where the keys are:

                                       'Curr'  : The disease state of this group of people [Mandatory]
                                                 (int where the disease-free state is 0 and the remaining
                                                 serotype colonisation states are positive integers up to 
                                                 the value set for 'num_of_serotypes' at class init)
                                       'npast' : Past number of colonisations for each serotype [Mandatory]
                                                 (array of length length 'num_of_serotypes' at class init)
                                        'Lam'  : The colonisation rate for each serotype [Mandatory]
                                                 (array of length length 'num_of_serotypes' at class init)
                                        'mu'   : The recovery rate from each serotype [Mandatory]
                                                 (array of length length 'num_of_serotypes' at class init)
                                         'f'   : The relative competitiveness of each serotype [Mandatory]
                                                 (array of length length 'num_of_serotypes' at class init)
                                        'eps'  : Nonspecific immunity (changes recovery rate) [Mandatory]
                                                 (positive float)
                                        'sig'  : Specific immunity (changes colonisation rate) [Mandatory]
                                                 (positive float)
                                       'mumax' : Maximum recovery rate from any serotype [Mandatory]
                                                 (positive float)
                                        'vef'  : Vaccine efficacy for each serotype (when applied) [Optional]
                                                 (array of length length 'num_of_serotypes' at class init)
                                        'vt'   : Vaccination times for each serotype [Optional]
                                                 (array of length length 'num_of_serotypes' at class init) 

        """

        # If the transmission rates are independent of the population state then setup the appropriate dictionaries
        if self.mode == 'fixed':

            # Extract the parameters and current disease state from the input
            vefs, vts = np.zeros(self.nsero), -np.ones(self.nsero)
            Curr, npasts, Lams = parameter_dic['Curr'], parameter_dic['npast'], parameter_dic['Lam']
            mus, fs, sig = parameter_dic['mu'], parameter_dic['f'], parameter_dic['sig']
            eps, mumax = parameter_dic['eps'], parameter_dic['mumax']
            if 'vef' in parameter_dic:
                vefs = parameter_dic['vef']
            if 'vt' in parameter_dic:
                vts = parameter_dic['vt']

            # Add to the total number of people in the class
            self.npeop += num_of_people

            # Set the immunity parameters of this population group
            self.pop['sig'] += [sig]*num_of_people
            self.pop['eps'] += [eps]*num_of_people
            self.pop['mumax'] += [mumax]*num_of_people
            self.ode_pop['sig'] += [sig]
            self.ode_pop['eps'] += [eps]
            self.ode_pop['mumax'] += [mumax]
            
            # Set the current disease state uniformally across this population group
            self.pop['Curr'] += [Curr]*num_of_people
            self.ode_pop['Curr'] += [Curr]

            # Set the population number for this group in the ODE system
            self.ode_pop['N'] += [num_of_people]

            # Loop over the number of serotypes to create all of the dictionary columns associated to this group of people
            for ns in range(1,self.nsero+1):   
                self.pop['npast_' + str(ns)] += [npasts[ns-1]]*num_of_people  
                self.pop['Lam_' + str(ns)] += [Lams[ns-1]]*num_of_people    
                self.pop['mu_' + str(ns)] += [mus[ns-1]]*num_of_people     
                self.pop['f_' + str(ns)] += [fs[ns-1]]*num_of_people
                self.pop['vef_' + str(ns)] += [vefs[ns-1]]*num_of_people
                self.pop['vt_' + str(ns)] += [vts[ns-1]]*num_of_people
                self.ode_pop['npast_' + str(ns)] += [npasts[ns-1]]
                self.ode_pop['Lam_' + str(ns)] += [Lams[ns-1]]    
                self.ode_pop['mu_' + str(ns)] += [mus[ns-1]]    
                self.ode_pop['f_' + str(ns)] += [fs[ns-1]]
                self.ode_pop['vef_' + str(ns)] += [vefs[ns-1]]
                self.ode_pop['vt_' + str(ns)] += [vts[ns-1]]


    def run_ode(self,runtime,timescale):
        """

        Method to run the corresponding ode system for the defined transmission model. The system generates
        all outputs through the pneumocode.ode_output dictionary.

        Args:

        [Mandatory] -   runtime    - The time period (in units of days) over which the system must run.

        [Mandatory] -  timescale   - The timescale (or stepsize) of the ode integrator.

        """

        # If the transmission rates are independent of the population state then run this ode system type
        if self.mode == 'fixed':

            # Extract the parameters in a form for faster simulation
            Ns = np.asarray(self.ode_pop['N'])
            num_of_groups = len(Ns)
            sigs, epss, mumaxs, Currs = np.asarray(self.ode_pop['sig']), np.asarray(self.ode_pop['eps']),\
                                        np.asarray(self.ode_pop['mumax']), np.asarray(self.ode_pop['Curr'])
            npasts, Lams, mus, fs, vefs, vts = [], [], [], [], [], []
            for ns in range(1,self.nsero+1):  
                npasts.append(self.ode_pop['npast_' + str(ns)])
                Lams.append(self.ode_pop['Lam_' + str(ns)])   
                mus.append(self.ode_pop['mu_' + str(ns)])    
                fs.append(self.ode_pop['f_' + str(ns)])
                vefs.append(self.ode_pop['vef_' + str(ns)])
                vts.append(self.ode_pop['vt_' + str(ns)])
            npasts, Lams, mus, fs = np.asarray(npasts), np.asarray(Lams), np.asarray(mus), np.asarray(fs)
            vefs, vts = np.asarray(vefs), np.asarray(vts)

            # Create higher-dimensional data structures for faster ode integration - index ordering is
            # typically: [serotype,group]
            groups_Currs = np.tensordot(np.ones(self.nsero),Currs,axes=0)
            groups_sigs = np.tensordot(np.ones(self.nsero),sigs,axes=0)
            groups_epss = np.tensordot(np.ones(self.nsero),epss,axes=0)
            groups_mumaxs = np.tensordot(np.ones(self.nsero),mumaxs,axes=0)
            groups_npasts, groups_Lams, groups_fs, groups_mus = npasts, Lams, fs, mus
            groups_vts, groups_vefs = vts, vefs
            groups_vefs_deliv = np.zeros((self.nsero,num_of_groups))

            # Define a function which takes the ode system forward in time by a step
            def next_step(qpn,t,dt=timescale):
                qpn_new = qpn
                q, p, n = qpn[0], qpn[1:self.nsero+1], qpn[self.nsero+1:2*self.nsero+1]
                due_mask = (t>=groups_vts)*(groups_vts!=-1.0)
                groups_vefs_deliv[due_mask] = groups_vefs[due_mask]
                groups_Lams_v = groups_Lams*(1.0-groups_vefs_deliv)
                groups_sigsLams_v = np.minimum(groups_sigs*groups_Lams,groups_Lams*(1.0-groups_vefs_deliv))
                n_new = n + dt*((groups_sigsLams_v*np.tensordot(np.ones(self.nsero),q,axes=0)) + \
                               ((np.tensordot(np.ones(self.nsero),np.sum(groups_fs*p,axis=0),axes=0)- \
                               (groups_fs*p))*groups_sigsLams_v))
                F = (groups_Lams_v*np.exp(-n)) + (groups_sigsLams_v*(1.0-np.exp(-n)))
                G = groups_mumaxs + (groups_mus-groups_mumaxs)*np.exp(np.tensordot(np.ones(self.nsero),\
                                                               np.sum(n,axis=0),axes=0)*(np.exp(-groups_epss)-1.0))
                q_new = q + dt*(np.sum(G*p,axis=0)-(np.sum(F,axis=0)*q))
                p_new = p + dt*((F*q)+((np.tensordot(np.ones(self.nsero),np.sum(groups_fs*p,axis=0),axes=0) - \
                                (groups_fs*p))*F)-(G*p)-((np.tensordot(np.ones(self.nsero),\
                                np.sum(F,axis=0),axes=0)-F)*groups_fs*p))
                qpn_new[0], qpn_new[1:self.nsero+1], qpn_new[self.nsero+1:2*self.nsero+1] = q_new, p_new, n_new 
                return qpn_new

            # Define a function which runs the system over the specified time period
            def run_system(qpn0,t0,t,dt=timescale):
                qpn = qpn0
                steps = int((t-t0)/dt)
                t = t0
                rec = []
                N_weights = np.tensordot(np.ones(2*self.nsero+1),Ns,axes=0)
                for i in range(0,steps):
                    qpn = next_step(qpn,t,dt)
                    outp = np.sum(N_weights*qpn,axis=1)/self.npeop
                    t += dt
                    rec.append(np.append(t,outp))
                rec = np.asarray(rec)
                ts, qs, ps, ns = rec[:,0], rec[:,1], rec[:,2:self.nsero+2], rec[:,self.nsero+2:2*self.nsero+2]
                return ts, qs, ps, ns

            # Run the system with consistent initial conditions and generate output dictionary
            q0 = np.zeros(num_of_groups)
            p0 = np.zeros((self.nsero,num_of_groups))
            n0 = groups_npasts
            q0[Currs==0] = 1.0
            p0[groups_Currs==np.tensordot(np.arange(1,self.nsero+1,1),np.ones(num_of_groups),axes=0)] = 1.0
            qpn0 = np.zeros((2*self.nsero+1,num_of_groups))
            qpn0[0] = q0
            qpn0[1:self.nsero+1] = p0
            qpn0[self.nsero+1:2*self.nsero+1] = n0
            t_vals, q_vals, p_vals, n_vals = run_system(qpn0,0,runtime)
            self.ode_output = {'time' : t_vals, 'probNone' : q_vals, 'probCurr' : p_vals, 'Expnpast' : n_vals}


    def run_sim(self,num_of_reals,runtime,timescale,time_snaps=[]):
        """

        Method to run a set of simulated realisations of the defined transmission model. The simulation generates
        all outputs through the pneumocode.sim_output dictionary.

        Args:

        [Mandatory] - num_of_reals - The number of independent realisations of the transmission process to be run.

        [Mandatory] -   runtime    - The time period (in units of days) over which the process must run.

        [Mandatory] -  timescale   - The timescale shorter than which it is safe to assume no events can occur.

        [Optional]  -  time_snaps  - If a list of times (in units of days) is given then the output will include
                                     snapshots of the population state at these points in time for all realisations
                                     in addition to the default output at the end of the 'runtime' period.

        """

        # Add the endpoint to the specified output
        time_snaps.append(runtime)

        # If the transmission rates are independent of the population state then run this simulation type
        if self.mode == 'fixed':

            # Extract the parameters in a form for faster simulation
            sigs, epss, mumaxs, Currs = np.asarray(self.pop['sig']), np.asarray(self.pop['eps']),\
                                        np.asarray(self.pop['mumax']), np.asarray(self.pop['Curr'])
            npasts, Lams, mus, fs, vefs, vts = [], [], [], [], [], []
            for ns in range(1,self.nsero+1):  
                npasts.append(self.pop['npast_' + str(ns)])
                Lams.append(self.pop['Lam_' + str(ns)])   
                mus.append(self.pop['mu_' + str(ns)])    
                fs.append(self.pop['f_' + str(ns)])
                vefs.append(self.pop['vef_' + str(ns)])
                vts.append(self.pop['vt_' + str(ns)])
            npasts, Lams, mus, fs = np.asarray(npasts), np.asarray(Lams), np.asarray(mus), np.asarray(fs)
            vefs, vts = np.asarray(vefs), np.asarray(vts)

            # Create higher-dimensional data structures for faster simulations - index ordering is
            # typically: [serotype,person,realisation]
            Currs_people = np.tensordot(Currs,np.ones(num_of_reals),axes=0)
            reals_Currs = np.tensordot(np.ones(self.nsero),Currs_people,axes=0)
            reals_sigs = np.tensordot(np.ones(self.nsero),np.tensordot(sigs,np.ones(num_of_reals),axes=0),axes=0)
            reals_epss = np.tensordot(np.ones(self.nsero),np.tensordot(epss,np.ones(num_of_reals),axes=0),axes=0)
            reals_mumaxs = np.tensordot(np.ones(self.nsero),np.tensordot(mumaxs,np.ones(num_of_reals),axes=0),axes=0)
            reals_npasts = np.tensordot(npasts,np.ones(num_of_reals),axes=0)
            reals_Lams = np.tensordot(Lams,np.ones(num_of_reals),axes=0)
            reals_mus = np.tensordot(mus,np.ones(num_of_reals),axes=0)
            reals_vts = np.tensordot(vts,np.ones(num_of_reals),axes=0)
            reals_vefs = np.tensordot(vefs,np.ones(num_of_reals),axes=0)
            reals_vefs_deliv = np.zeros((self.nsero,self.npeop,num_of_reals))

            # Create output array storage for fast updates
            Curr_store = np.zeros((len(time_snaps),self.npeop,num_of_reals))
            npast_store = np.zeros((len(time_snaps),self.nsero,num_of_reals))

            # Initialise the loop over realisations in time
            times = np.zeros(num_of_reals)
            slowest_time = 0.0
            still_running = np.ones(num_of_reals)
            while slowest_time < runtime:

                # Store the previous times before step
                previous_times = times.copy()

                # Change the indicator for the realisations which have ended
                still_running[times>runtime] = 0.0

                # Draw the next point in time
                timestep = np.random.exponential(timescale,size=num_of_reals)
                times += still_running*timestep

                # Find the slowest realisation
                slowest_time = np.ndarray.min(times)

                # Draw event realisations for each person
                events = np.random.uniform(size=(self.npeop,num_of_reals))

                # Work out whether vaccinations are due
                due_mask = (np.tensordot(np.ones((self.nsero,self.npeop)),\
                            times,axes=0)>=reals_vts)*(reals_vts!=-1.0)
                reals_vefs_deliv[due_mask] = reals_vefs[due_mask]

                # Create cumulative rate sums that are consistent with the present state
                sum_reals_npasts = np.tensordot(np.ones(self.nsero),np.sum(reals_npasts,axis=0),axes=0)
                rec_rates = reals_mumaxs+((reals_mus-reals_mumaxs)*np.exp(-reals_epss*sum_reals_npasts))
                col_rates = reals_Lams*np.minimum(((reals_npasts==0)+\
                                                   (reals_sigs*(reals_npasts>0))),1.0-reals_vefs_deliv)
                reals_fs = np.tensordot(np.ones(self.nsero),\
                           np.diagonal(fs[Currs_people.astype(int)-1],axis1=0,axis2=2).T,axes=0)
                ccl_rates = col_rates*reals_fs
                rec_rates[(reals_Currs==0) | (reals_Currs!=np.tensordot(np.arange(1,self.nsero+1,1),\
                                                           np.ones((self.npeop,num_of_reals)),axes=0))] = 0.0
                col_rates[(reals_Currs>0)] = 0.0
                ccl_rates[(reals_Currs==0) | (reals_Currs==np.tensordot(np.arange(1,self.nsero+1,1),\
                                                           np.ones((self.npeop,num_of_reals)),axes=0))] = 0.0
                cumsum_rec_rates = np.cumsum(rec_rates,axis=0)
                cumsum_col_rates = np.cumsum(col_rates,axis=0)
                cumsum_ccl_rates = np.cumsum(ccl_rates,axis=0)

                # Store the previous states before they are changed
                previous_Currs_people = Currs_people.copy()

                # Use the event realisations and the cumulative rate sums to evaluate the next state transitions
                reals_tot_rate_sums = np.tensordot(np.ones((self.nsero,self.npeop)),1.0/timestep,axes=0) + \
                                     (np.tensordot(np.ones(self.nsero),cumsum_rec_rates[-1],axes=0)*(reals_Currs>0)) + \
                                     (np.tensordot(np.ones(self.nsero),cumsum_col_rates[-1],axes=0)*(reals_Currs==0)) + \
                                     (np.tensordot(np.ones(self.nsero),cumsum_ccl_rates[-1],axes=0)*(reals_Currs>0))
                reals_events = np.tensordot(np.ones(self.nsero),events,axes=0)
                Currs_people[np.tensordot(np.ones(self.npeop),still_running,axes=0)==1.0] = \
                                     (0 + np.sum(np.tensordot(np.arange(1,self.nsero+1,1),\
                                     np.ones((self.npeop,num_of_reals)),axes=0)*\
                                     (((np.append(np.zeros((1,self.npeop,num_of_reals)),\
                                     cumsum_col_rates[:-1],axis=0)/reals_tot_rate_sums<reals_events) & \
                                     (reals_events<=cumsum_col_rates/reals_tot_rate_sums)) + \
                                     ((np.append(np.zeros((1,self.npeop,num_of_reals)),\
                                     cumsum_ccl_rates[:-1],axis=0)/reals_tot_rate_sums<reals_events) & \
                                     (reals_events<=cumsum_ccl_rates/reals_tot_rate_sums))),axis=0) + \
                                     Currs_people*((cumsum_ccl_rates[-1]+cumsum_rec_rates[-1])/\
                                     reals_tot_rate_sums[0]<events))[np.tensordot(np.ones(self.npeop),\
                                     still_running,axes=0)==1.0]

                # Update the number of past colonisations of each individual to match the new states
                reals_Currs = np.tensordot(np.ones(self.nsero),Currs_people,axes=0)
                previous_reals_Currs = np.tensordot(np.ones(self.nsero),previous_Currs_people,axes=0)
                reals_npasts = reals_npasts + ((reals_Currs!=previous_reals_Currs) & (reals_Currs!=0))*\
                                              (np.tensordot(np.arange(1,self.nsero+1,1),\
                                              np.ones((self.npeop,num_of_reals)),axes=0)==reals_Currs)

                # Use the previous times and the time_snaps list to append output with relevant information
                Curr_store += (np.tensordot(np.ones((len(time_snaps),self.npeop)),times,axes=0)>\
                              np.tensordot(np.asarray(time_snaps),np.ones((self.npeop,num_of_reals)),axes=0))*\
                              (np.tensordot(np.asarray(time_snaps),np.ones((self.npeop,num_of_reals)),axes=0)>=\
                              np.tensordot(np.ones((len(time_snaps),self.npeop)),previous_times,axes=0))*\
                              np.tensordot(np.ones(len(time_snaps)),Currs_people,axes=0)
                npast_store += (np.tensordot(np.ones((len(time_snaps),self.nsero)),times,axes=0)>\
                               np.tensordot(np.asarray(time_snaps),np.ones((self.nsero,num_of_reals)),axes=0))*\
                               (np.tensordot(np.asarray(time_snaps),np.ones((self.nsero,num_of_reals)),axes=0)>=\
                               np.tensordot(np.ones((len(time_snaps),self.nsero)),previous_times,axes=0))*\
                               np.tensordot(np.ones(len(time_snaps)),np.sum(reals_npasts,axis=1),axes=0)                 

            # Create simulation output
            self.sim_output = {'Curr' : {time_snaps[ti] : Curr_store[ti] for ti in range(0,len(time_snaps))},\
                               'npastsum' : {time_snaps[ti] : npast_store[ti] for ti in range(0,len(time_snaps))}}  


    def fit_data(self,df):
        """

        Method to fit the maximum a posteriori parameters to an input population dataset.

        Args:

        [Mandatory] -   df   - A pandas DataFrame of observations with the following columns: 'person_id', 'time',
                                'serotype_id'. Where the time column is in units of days.


        """

        print('Hello')
            
