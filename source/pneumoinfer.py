"""

pneumoinfer - A stochastic simulation and ODE solver
class for inferring multi-state models with a counting 
memory which records previous state occupations. This
is currently written for pneumococcus-like models but 
may be generalised to others in the future.

This is the source code for the class in which you can 
find comments on how pneumoinfer works internally. Please 
also feel free to consult the Jupyter Notebooks in the
public repo for a more user-friendly experience.

"""

import numpy as np
import pandas as pd
import scipy.special as spec


# Initialise the 'pneumoinfer' method class
class pneumoinfer:
    def __init__(self, num_of_states: int):
        """

        pneumoinfer - An all-in-one inference and stochastic 
        simulation class for multi-state models with a counting 
        memory which records previous state occupations.

        Args:
        num_of_states
            The effective number of states (separate states) 
            that each individual member can exist in. This does 
            not include the null state.
     
        """

        # Set initialisation variables of the class
        self.nstat = num_of_states
        self.nind = 0
        self._pop = None
        self._ode_pop = None
        self._cont_mat = None

    @property
    def pop(self):
        if self._pop is None:
            self._pop = {"sig": [], "eps": [], "mumax": [], "Curr": []}
            for ns in range(1, self.nstat + 1):
                # The number of past occupations of this state
                self._pop["npast_" + str(ns)] = []
                # The recovery rate from this state
                self._pop["mu_" + str(ns)] = []
                # The competitiveness factor of this state
                self._pop["f_" + str(ns)] = []
                # The vaccine efficacy against this state
                self._pop["vef_" + str(ns)] = []
                # The time of vaccination against this state
                self._pop["vt_" + str(ns)] = []
                # The occupation rate of this state
                # (or its minimum in "vary" mode)
                self._pop["Lam_" + str(ns)] = []
        return self._pop

    @property
    def ode_pop(self):
        if self._ode_pop is None:
            self._ode_pop = {
                "sig": [], 
                "eps": [], 
                "mumax": [], 
                "Curr": [], 
                "N": [], 
                "dataCount" : [],
                "dataTime" : [],
                "dataCurr" : [],
            }
            for ns in range(1, self.nstat + 1):
                self._ode_pop["npast_" + str(ns)] = []
                self._ode_pop["mu_" + str(ns)] = []
                self._ode_pop["f_" + str(ns)] = []
                self._ode_pop["vef_" + str(ns)] = []
                self._ode_pop["vt_" + str(ns)] = []
                self._ode_pop["Lam_" + str(ns)] = []
        return self._ode_pop

    @property
    def cont_mat(self):
        """

        Including or not including a contact matrix determines the mode for the 
        modelling approach. If self.mode = "fixed", the occupation rates are 
        independent of the ensemble state. If self.mode = "vary", the occupation 
        rates depend on the ensemble state.

        """
        if self._cont_mat is None:
            self.mode = "fixed"
        else:
            self.mode = "vary"
        return self._cont_mat

    def create_members(
        self, 
        num_of_members: int, 
        parameter_dic: dict,
        data: pd.DataFrame = None,
    ):
        """

        Method to add members (or a single member) to the ensemble.

        Args:
        num_of_members
            The number of members in this added group to the ensemble.

        parameter_dic
            A dictionary of parameters and parameter arrays where the keys are:

                'Curr'  : The state of this group of individuals [Mandatory]
                (int where the null state is 0 and the remaining states are
                positive integers up to the value set for 'num_of_states' at 
                class init)

                'npast' : Past number of occupations for each state [Mandatory]
                (array of length length 'num_of_states' at class init)
                
                'Lam'  : The occupation rate for each state if no contact
                matrix is included OR the minimum occupation rate for this
                state if there is a contact matrix included [Mandatory]
                (array of length length 'num_of_states' at class init)

                'mu'   : The recovery rate from each state [Mandatory]
                (array of length length 'num_of_states' at class init)

                'f'   : The relative competitiveness of each state [Mandatory]
                (array of length length 'num_of_states' at class init)

                'eps'  : Nonspecific memory (changes recovery rate) [Mandatory]
                (positive float)

                'sig'  : Specific memory (changes occupation rate) [Mandatory]
                (positive float)

                'mumax' : Maximum recovery rate from any state [Mandatory]
                (positive float)

                'vef'  : Vaccine efficacy for each state (when applied) [Optional]
                (array of length length 'num_of_states' at class init)

                'vt'   : Vaccination times for each state [Optional]
                (array of length length 'num_of_states' at class init)

                'cind' : 0-axis index of this group of individuals with 
                respect to the contact matrix (if included) [Optional]
                (int with max value less than the 0-axis length of the 
                contact matrix)

        Keywords:
        data
            An optional pandas DataFrame of state observations at different
            time points which must be associated to a count of individuals 
            in the ensemble. The DataFrame must have the following columns:

                'Count' : The number of individuals associated to this recorded
                observation at this point in time [Mandatory]
                (positive int)

                'Time' : Time of the observation [Mandatory]
                (float value beginning at 0.0 in units corresponding 
                to the desired rates)

                'Curr' : The state this individual was observed in [Mandatory]
                (positive int where 0 is the null state)

        """

        # Initialise the contact matrix property (must be included
        # before creating a population) and then add an index to it
        # in the population definition, if necessary
        cont_mat = self.cont_mat
        if self.mode == "vary":
            if self._pop is None:
                pop = self.pop
                self._pop["cind"] = []
            if self._ode_pop is None:
                ode_pop = self.ode_pop
                self._ode_pop["cind"] = []
            cind = parameter_dic["cind"]
            self.pop["cind"] += [cind] * num_of_members
            self.ode_pop["cind"] += [cind]

        # Extract the parameters and current state from the input
        vefs, vts = np.zeros(self.nstat), -np.ones(self.nstat)
        Curr, npasts, Lams = (
            parameter_dic["Curr"],
            parameter_dic["npast"],
            parameter_dic["Lam"],
        )
        mus, fs, sig = parameter_dic["mu"], parameter_dic["f"], parameter_dic["sig"]
        eps, mumax = parameter_dic["eps"], parameter_dic["mumax"]
        if "vef" in parameter_dic:
            vefs = parameter_dic["vef"]
        if "vt" in parameter_dic:
            vts = parameter_dic["vt"]

        # Add to the total number of individuals in the ensemble
        self.nind += num_of_members

        # Set the memory parameters of this ensemble group
        self.pop["sig"] += [sig] * num_of_members
        self.pop["eps"] += [eps] * num_of_members
        self.pop["mumax"] += [mumax] * num_of_members
        self.ode_pop["sig"] += [sig]
        self.ode_pop["eps"] += [eps]
        self.ode_pop["mumax"] += [mumax]

        # Set the current state uniformally across this ensemble group
        self.pop["Curr"] += [Curr] * num_of_members
        self.ode_pop["Curr"] += [Curr]

        # If data has been added then include it with the ODE ensemble
        if data is not None:
            self.ode_pop["dataCount"] += [data.Count.values]
            self.ode_pop["dataTime"] += [data.Time.values]
            self.ode_pop["dataCurr"] += [data.Curr.values]
        else:
            self.ode_pop["dataCount"] += [None]
            self.ode_pop["dataTime"] += [None]
            self.ode_pop["dataCurr"] += [None]

        # Set the ensemble size for this group in the ODE system
        self.ode_pop["N"] += [num_of_members]

        # Loop over the number of states to create all of the dictionary
        # columns associated to this group of individuals
        for ns in range(1, self.nstat + 1):
            self.pop["npast_" + str(ns)] += [npasts[ns - 1]] * num_of_members
            self.pop["Lam_" + str(ns)] += [Lams[ns - 1]] * num_of_members
            self.pop["mu_" + str(ns)] += [mus[ns - 1]] * num_of_members
            self.pop["f_" + str(ns)] += [fs[ns - 1]] * num_of_members
            self.pop["vef_" + str(ns)] += [vefs[ns - 1]] * num_of_members
            self.pop["vt_" + str(ns)] += [vts[ns - 1]] * num_of_members
            self.ode_pop["npast_" + str(ns)] += [npasts[ns - 1]]
            self.ode_pop["Lam_" + str(ns)] += [Lams[ns - 1]]
            self.ode_pop["mu_" + str(ns)] += [mus[ns - 1]]
            self.ode_pop["f_" + str(ns)] += [fs[ns - 1]]
            self.ode_pop["vef_" + str(ns)] += [vefs[ns - 1]]
            self.ode_pop["vt_" + str(ns)] += [vts[ns - 1]]

    def run_ode(self, runtime: float, timescale: float):
        """

        Method to run the corresponding ode system for the
        defined occupation model. The system generates
        all outputs through the pneumoinfer.ode_output dictionary.

        Args:
        runtime
            The time period (in units of days) over which the 
            system must run.
        timescale
            The timescale (or stepsize) of the ode integrator.

        """

        # Extract the parameters in a form for faster simulation
        Ns = np.asarray(self.ode_pop["N"])
        num_of_groups = len(Ns)
        sigs, epss, mumaxs, Currs = (
            np.asarray(self.ode_pop["sig"]),
            np.asarray(self.ode_pop["eps"]),
            np.asarray(self.ode_pop["mumax"]),
            np.asarray(self.ode_pop["Curr"]),
        )
        npasts, Lams, mus, fs, vefs, vts = [], [], [], [], [], []
        for ns in range(1, self.nstat + 1):
            npasts.append(self.ode_pop["npast_" + str(ns)])
            Lams.append(self.ode_pop["Lam_" + str(ns)])
            mus.append(self.ode_pop["mu_" + str(ns)])
            fs.append(self.ode_pop["f_" + str(ns)])
            vefs.append(self.ode_pop["vef_" + str(ns)])
            vts.append(self.ode_pop["vt_" + str(ns)])
        npasts, Lams, mus, fs = (
            np.asarray(npasts),
            np.asarray(Lams),
            np.asarray(mus),
            np.asarray(fs),
        )
        vefs, vts = np.asarray(vefs), np.asarray(vts)

        # Create higher-dimensional data structures for faster
        # ode integration - index ordering is typically: [state,group]
        groups_Currs = np.tensordot(np.ones(self.nstat), Currs, axes=0)
        groups_sigs = np.tensordot(np.ones(self.nstat), sigs, axes=0)
        groups_epss = np.tensordot(np.ones(self.nstat), epss, axes=0)
        groups_mumaxs = np.tensordot(np.ones(self.nstat), mumaxs, axes=0)
        groups_npasts, groups_Lams, groups_fs, groups_mus = npasts, Lams, fs, mus
        groups_vts, groups_vefs = vts, vefs
        groups_vefs_deliv = np.zeros((self.nstat, num_of_groups))

        # Get the contact matrix if it exists
        cont_mat = self.cont_mat

        # If the occupation rates are independent of the ensemble
        # state then run this ode system type
        if self.mode == "fixed":

            def Lams_function(p, groups_Lams):
                return groups_Lams

        # If the occupation rates are dependent on the ensemble
        # state then run this ode system type instead
        if self.mode == "vary":
            cinds = np.asarray(self.ode_pop["cind"])
            ind_contact = []
            for ci in cinds:
                ind_contact.append(cont_mat[ci][cinds])
            ind_contact = np.asarray(ind_contact)
            totN = np.sum(Ns)

            def Lams_function(p, groups_Lams):
                contp = np.tensordot(ind_contact, p, axes=([1], [1])).swapaxes(0, 1)
                return groups_Lams + (contp * Ns / totN)

        # Define a function which takes the ode system forward
        # in time by a step
        def next_step(qpn, t, dt=timescale):
            qpn_new = qpn
            q, p, n = (
                qpn[0],
                qpn[1 : self.nstat + 1],
                qpn[self.nstat + 1 : 2 * self.nstat + 1],
            )
            due_mask = (t >= groups_vts) * (groups_vts != -1.0)
            groups_vefs_deliv[due_mask] = groups_vefs[due_mask]
            gLs = Lams_function(p, groups_Lams)
            groups_Lams_v = gLs * (1.0 - groups_vefs_deliv)
            groups_sigsLams_v = np.minimum(
                groups_sigs * gLs, gLs * (1.0 - groups_vefs_deliv)
            )
            n_new = n + dt * (
                (groups_sigsLams_v * np.tensordot(np.ones(self.nstat), q, axes=0))
                + (
                    (
                        np.tensordot(
                            np.ones(self.nstat), np.sum(groups_fs * p, axis=0), axes=0,
                        )
                        - (groups_fs * p)
                    )
                    * groups_sigsLams_v
                )
            )
            F = (groups_Lams_v * np.exp(-n)) + (groups_sigsLams_v * (1.0 - np.exp(-n)))
            G = groups_mumaxs + (groups_mus - groups_mumaxs) * np.exp(
                np.tensordot(np.ones(self.nstat), np.sum(n, axis=0), axes=0)
                * (np.exp(-groups_epss) - 1.0)
            )
            q_new = q + dt * (np.sum(G * p, axis=0) - (np.sum(F, axis=0) * q))
            p_new = p + dt * (
                (F * q)
                + (
                    (
                        np.tensordot(
                            np.ones(self.nstat), np.sum(groups_fs * p, axis=0), axes=0,
                        )
                        - (groups_fs * p)
                    )
                    * F
                )
                - (G * p)
                - (
                    (np.tensordot(np.ones(self.nstat), np.sum(F, axis=0), axes=0) - F)
                    * groups_fs
                    * p
                )
            )
            (
                qpn_new[0],
                qpn_new[1 : self.nstat + 1],
                qpn_new[self.nstat + 1 : 2 * self.nstat + 1],
            ) = (q_new, p_new, n_new)
            return qpn_new

        # Define a function which runs the system over the specified
        # time period
        def run_system(qpn0, t0, tend, dt=timescale):
            qpn = qpn0
            steps = int((tend - t0) / dt)
            t = t0
            rec = []
            N_weights = np.tensordot(np.ones(2 * self.nstat + 1), Ns, axes=0)
            for i in range(0, steps):
                qpn = next_step(qpn, t, dt=dt)
                outp = np.sum(N_weights * qpn, axis=1) / self.nind
                t += dt
                rec.append(np.append(t, outp))
            rec = np.asarray(rec)
            ts, qs, ps, ns = (
                rec[:, 0],
                rec[:, 1],
                rec[:, 2 : self.nstat + 2],
                rec[:, self.nstat + 2 : 2 * self.nstat + 2],
            )
            return ts, qs, ps, ns

        # Run the system with consistent initial conditions and generate
        # output dictionary
        q0 = np.zeros(num_of_groups)
        p0 = np.zeros((self.nstat, num_of_groups))
        n0 = groups_npasts
        q0[Currs == 0] = 1.0
        p0[
            groups_Currs
            == np.tensordot(
                np.arange(1, self.nstat + 1, 1), np.ones(num_of_groups), axes=0
            )
        ] = 1.0
        qpn0 = np.zeros((2 * self.nstat + 1, num_of_groups))
        qpn0[0] = q0
        qpn0[1 : self.nstat + 1] = p0
        qpn0[self.nstat + 1 : 2 * self.nstat + 1] = n0
        t_vals, q_vals, p_vals, n_vals = run_system(qpn0, 0, runtime)
        self.ode_output = {
            "time": t_vals,
            "probNone": q_vals,
            "probCurr": p_vals,
            "Expnpast": n_vals,
        }

    def run_sim(
        self, num_of_reals: int, runtime: float, timescale: float, time_snaps=[]
    ):
        """

        Method to run a set of simulated realisations of the defined 
        occupation model. The simulation stores all outputs through 
        the pneumoinfer.sim_output dictionary.

        Args:
        num_of_reals
            The number of independent realisations of the occupation 
            process to be run.
        runtime
            The time period (in units of days) over which the process 
            must run.
        timescale
            The timescale shorter than which it is safe to assume 
            no events can occur.
        time_snaps
            If a list of times (in units of days) is given then the 
            output will include snapshots of the ensemble state at 
            these points in time for all realisations in addition to the 
            default output at the end of the 'runtime' period.

        """

        # Add the endpoint to the specified output
        time_snaps.append(runtime)

        # Extract the parameters in a form for faster simulation
        sigs, epss, mumaxs, Currs = (
            np.asarray(self.pop["sig"]),
            np.asarray(self.pop["eps"]),
            np.asarray(self.pop["mumax"]),
            np.asarray(self.pop["Curr"]),
        )
        npasts, Lams, mus, fs, vefs, vts = [], [], [], [], [], []
        for ns in range(1, self.nstat + 1):
            npasts.append(self.pop["npast_" + str(ns)])
            Lams.append(self.pop["Lam_" + str(ns)])
            mus.append(self.pop["mu_" + str(ns)])
            fs.append(self.pop["f_" + str(ns)])
            vefs.append(self.pop["vef_" + str(ns)])
            vts.append(self.pop["vt_" + str(ns)])
        npasts, Lams, mus, fs = (
            np.asarray(npasts),
            np.asarray(Lams),
            np.asarray(mus),
            np.asarray(fs),
        )
        vefs, vts = np.asarray(vefs), np.asarray(vts)

        # Create higher-dimensional data structures for faster
        # simulations - index ordering is typically:
        # [state,individual,realisation]
        Currs_members = np.tensordot(Currs, np.ones(num_of_reals), axes=0)
        reals_Currs = np.tensordot(np.ones(self.nstat), Currs_members, axes=0)
        reals_sigs = np.tensordot(
            np.ones(self.nstat),
            np.tensordot(sigs, np.ones(num_of_reals), axes=0),
            axes=0,
        )
        reals_epss = np.tensordot(
            np.ones(self.nstat),
            np.tensordot(epss, np.ones(num_of_reals), axes=0),
            axes=0,
        )
        reals_mumaxs = np.tensordot(
            np.ones(self.nstat),
            np.tensordot(mumaxs, np.ones(num_of_reals), axes=0),
            axes=0,
        )
        reals_npasts = np.tensordot(npasts, np.ones(num_of_reals), axes=0)
        reals_Lams = np.tensordot(Lams, np.ones(num_of_reals), axes=0)
        reals_mus = np.tensordot(mus, np.ones(num_of_reals), axes=0)
        reals_vts = np.tensordot(vts, np.ones(num_of_reals), axes=0)
        reals_vefs = np.tensordot(vefs, np.ones(num_of_reals), axes=0)
        reals_vefs_deliv = np.zeros((self.nstat, self.nind, num_of_reals))

        # Create output array storage for fast updates
        Curr_store = np.zeros((len(time_snaps), self.nind, num_of_reals))
        npast_store = np.zeros((len(time_snaps), self.nstat, num_of_reals))

        # Get the contact matrix if it exists
        cont_mat = self.cont_mat

        # If the occupation rates are independent of
        # the ensemble state then run this simulation type
        if self.mode == "fixed":

            def col_rates_func(r_Currs, r_npasts, r_vefs_deliv):
                return reals_Lams * np.minimum(
                    ((r_npasts == 0) + (reals_sigs * (r_npasts > 0))),
                    1.0 - r_vefs_deliv,
                )

        # If the occupation rates are instead dependent on
        # the ensemble state then run this simulation type
        if self.mode == "vary":
            cinds = np.asarray(self.pop["cind"])
            ind_contact = []
            for ci in cinds:
                ind_contact.append(cont_mat[ci][cinds])
            ind_contact = np.asarray(ind_contact)
            r_Currs_ref = np.tensordot(
                np.arange(1, self.nstat + 1, 1),
                np.ones((self.nind, num_of_reals)),
                axes=0,
            )

            def col_rates_func(r_Currs, r_npasts, r_vefs_deliv):
                reals_contact_Lams = reals_Lams + (
                    np.tensordot(
                        ind_contact, (r_Currs_ref == r_Currs), axes=([1], [1]),
                    ).swapaxes(0, 1)
                    / float(self.nind)
                )
                return reals_contact_Lams * np.minimum(
                    ((r_npasts == 0) + (reals_sigs * (r_npasts > 0))),
                    1.0 - r_vefs_deliv,
                )

        # Initialise the loop over realisations in time
        times = np.zeros(num_of_reals)
        slowest_time = 0.0
        still_running = np.ones(num_of_reals)
        while slowest_time < runtime:

            # Store the previous times before step
            previous_times = times.copy()

            # Change the indicator for the realisations which
            # have ended
            still_running[times > runtime] = 0.0

            # Draw the next point in time
            timestep = np.random.exponential(timescale, size=num_of_reals)
            times += still_running * timestep

            # Find the slowest realisation
            slowest_time = np.ndarray.min(times)

            # Draw event realisations for each individual member
            # of the ensemble
            events = np.random.uniform(size=(self.nind, num_of_reals))

            # Work out whether vaccinations are due
            due_mask = (
                np.tensordot(np.ones((self.nstat, self.nind)), times, axes=0)
                >= reals_vts
            ) * (reals_vts != -1.0)
            reals_vefs_deliv[due_mask] = reals_vefs[due_mask]

            # Create cumulative rate sums that are consistent with
            # the present state
            sum_reals_npasts = np.tensordot(
                np.ones(self.nstat), np.sum(reals_npasts, axis=0), axes=0
            )
            rec_rates = reals_mumaxs + (
                (reals_mus - reals_mumaxs) * np.exp(-reals_epss * sum_reals_npasts)
            )
            col_rates = col_rates_func(reals_Currs, reals_npasts, reals_vefs_deliv)
            reals_fs = np.tensordot(
                np.ones(self.nstat),
                np.diagonal(fs[Currs_members.astype(int) - 1], axis1=0, axis2=2).T,
                axes=0,
            )
            ccl_rates = col_rates * reals_fs
            rec_rates[
                (reals_Currs == 0)
                | (
                    reals_Currs
                    != np.tensordot(
                        np.arange(1, self.nstat + 1, 1),
                        np.ones((self.nind, num_of_reals)),
                        axes=0,
                    )
                )
            ] = 0.0
            col_rates[(reals_Currs > 0)] = 0.0
            ccl_rates[
                (reals_Currs == 0)
                | (
                    reals_Currs
                    == np.tensordot(
                        np.arange(1, self.nstat + 1, 1),
                        np.ones((self.nind, num_of_reals)),
                        axes=0,
                    )
                )
            ] = 0.0
            cumsum_rec_rates = np.cumsum(rec_rates, axis=0)
            cumsum_col_rates = np.cumsum(col_rates, axis=0)
            cumsum_ccl_rates = np.cumsum(ccl_rates, axis=0)

            # Store the previous states before they are changed
            previous_Currs_members = Currs_members.copy()

            # Use the event realisations and the cumulative rate
            # sums to evaluate the next state transitions
            reals_tot_rate_sums = (
                np.tensordot(np.ones((self.nstat, self.nind)), 1.0 / timestep, axes=0)
                + (
                    np.tensordot(np.ones(self.nstat), cumsum_rec_rates[-1], axes=0)
                    * (reals_Currs > 0)
                )
                + (
                    np.tensordot(np.ones(self.nstat), cumsum_col_rates[-1], axes=0)
                    * (reals_Currs == 0)
                )
                + (
                    np.tensordot(np.ones(self.nstat), cumsum_ccl_rates[-1], axes=0)
                    * (reals_Currs > 0)
                )
            )
            reals_events = np.tensordot(np.ones(self.nstat), events, axes=0)
            Currs_members[
                np.tensordot(np.ones(self.nind), still_running, axes=0) == 1.0
            ] = (
                0
                + np.sum(
                    np.tensordot(
                        np.arange(1, self.nstat + 1, 1),
                        np.ones((self.nind, num_of_reals)),
                        axes=0,
                    )
                    * (
                        (
                            (
                                np.append(
                                    np.zeros((1, self.nind, num_of_reals)),
                                    cumsum_col_rates[:-1],
                                    axis=0,
                                )
                                / reals_tot_rate_sums
                                < reals_events
                            )
                            & (reals_events <= cumsum_col_rates / reals_tot_rate_sums)
                        )
                        + (
                            (
                                np.append(
                                    np.zeros((1, self.nind, num_of_reals)),
                                    cumsum_ccl_rates[:-1],
                                    axis=0,
                                )
                                / reals_tot_rate_sums
                                < reals_events
                            )
                            & (reals_events <= cumsum_ccl_rates / reals_tot_rate_sums)
                        )
                    ),
                    axis=0,
                )
                + Currs_members
                * (
                    (cumsum_ccl_rates[-1] + cumsum_rec_rates[-1])
                    / reals_tot_rate_sums[0]
                    < events
                )
            )[
                np.tensordot(np.ones(self.nind), still_running, axes=0) == 1.0
            ]

            # Update the number of past occupations of each individual
            # to match the new states
            reals_Currs = np.tensordot(np.ones(self.nstat), Currs_members, axes=0)
            previous_reals_Currs = np.tensordot(
                np.ones(self.nstat), previous_Currs_members, axes=0
            )
            reals_npasts = reals_npasts + (
                (reals_Currs != previous_reals_Currs) & (reals_Currs != 0)
            ) * (
                np.tensordot(
                    np.arange(1, self.nstat + 1, 1),
                    np.ones((self.nind, num_of_reals)),
                    axes=0,
                )
                == reals_Currs
            )

            # Use the previous times and the time_snaps list to append
            # output with relevant information
            Curr_store += (
                (
                    np.tensordot(np.ones((len(time_snaps), self.nind)), times, axes=0)
                    > np.tensordot(
                        np.asarray(time_snaps),
                        np.ones((self.nind, num_of_reals)),
                        axes=0,
                    )
                )
                * (
                    np.tensordot(
                        np.asarray(time_snaps),
                        np.ones((self.nind, num_of_reals)),
                        axes=0,
                    )
                    >= np.tensordot(
                        np.ones((len(time_snaps), self.nind)), previous_times, axes=0,
                    )
                )
                * np.tensordot(np.ones(len(time_snaps)), Currs_members, axes=0)
            )
            npast_store += (
                (
                    np.tensordot(np.ones((len(time_snaps), self.nstat)), times, axes=0)
                    > np.tensordot(
                        np.asarray(time_snaps),
                        np.ones((self.nstat, num_of_reals)),
                        axes=0,
                    )
                )
                * (
                    np.tensordot(
                        np.asarray(time_snaps),
                        np.ones((self.nstat, num_of_reals)),
                        axes=0,
                    )
                    >= np.tensordot(
                        np.ones((len(time_snaps), self.nstat)), previous_times, axes=0,
                    )
                )
                * np.tensordot(
                    np.ones(len(time_snaps)), np.sum(reals_npasts, axis=1), axes=0
                )
            )

            # Create simulation output
            self.sim_output = {
                "Curr": {
                    time_snaps[ti]: Curr_store[ti] for ti in range(0, len(time_snaps))
                },
                "npastsum": {
                    time_snaps[ti]: npast_store[ti] for ti in range(0, len(time_snaps))
                },
            }

    def maximise_lnlike(self, timescale: float) -> float:
        """

        Method maximise the log-likelihood given the input set of data
        via the 'create_members' method. This makes use of the ODE 
        approximation to the full simulation to evaluate the log-likelihood
        and its gradient (the latter is computed using a 'multiple adjoint'
        method inspired by https://arxiv.org/abs/2006.02493).

        Args:
        timescale
            The timescale (or stepsize) of the ode integrator.

        """
        
        # Extract the parameters in a form for faster simulation
        Ns = np.asarray(self.ode_pop["N"])
        num_of_groups = len(Ns)
        sigs, epss, mumaxs, Currs = (
            np.asarray(self.ode_pop["sig"]),
            np.asarray(self.ode_pop["eps"]),
            np.asarray(self.ode_pop["mumax"]),
            np.asarray(self.ode_pop["Curr"]),
        )
        npasts, Lams, mus, fs, vefs, vts = [], [], [], [], [], []
        for ns in range(1, self.nstat + 1):
            npasts.append(self.ode_pop["npast_" + str(ns)])
            Lams.append(self.ode_pop["Lam_" + str(ns)])
            mus.append(self.ode_pop["mu_" + str(ns)])
            fs.append(self.ode_pop["f_" + str(ns)])
            vefs.append(self.ode_pop["vef_" + str(ns)])
            vts.append(self.ode_pop["vt_" + str(ns)])
        npasts, Lams, mus, fs = (
            np.asarray(npasts),
            np.asarray(Lams),
            np.asarray(mus),
            np.asarray(fs),
        )
        vefs, vts = np.asarray(vefs), np.asarray(vts)

        # Construct data vector representation
        data_groups = []
        gid = 0
        for v in self.ode_pop["dataCount"]:
            if v is not None:
                data_groups.append([gid]*len(v))
            gid += 1
        data_counts = np.array([])
        data_times = np.array([])
        data_Currs = np.array([])
        for vi in range(0, len(self.ode_pop["dataCount"])):
            co = self.ode_pop["dataCount"][vi]
            cu = self.ode_pop["dataCurr"][vi]
            ti = self.ode_pop["dataTime"][vi]
            if co is not None:
                data_counts = np.append(data_counts, co)
            if cu is not None:
                data_Currs = np.append(data_Currs, cu)
            if ti is not None:
                data_times = np.append(data_times, ti)
        data_groups = np.asarray(data_groups).flatten().astype(int)
        data_Currs = data_Currs.astype(int)

        # Create higher-dimensional data structures for faster
        # ode integration - index ordering is typically: [state,group]
        groups_Currs = np.tensordot(np.ones(self.nstat), Currs, axes=0)
        groups_sigs = np.tensordot(np.ones(self.nstat), sigs, axes=0)
        groups_epss = np.tensordot(np.ones(self.nstat), epss, axes=0)
        groups_mumaxs = np.tensordot(np.ones(self.nstat), mumaxs, axes=0)
        groups_npasts, groups_Lams, groups_fs, groups_mus = npasts, Lams, fs, mus
        groups_vts, groups_vefs = vts, vefs
        groups_vefs_deliv = np.zeros((self.nstat, num_of_groups))

        # Get the contact matrix if it exists
        cont_mat = self.cont_mat

        # If the occupation rates are independent of the ensemble
        # state then run this ode system type
        if self.mode == "fixed":

            def Lams_function(p, groups_Lams):
                return groups_Lams

        # If the occupation rates are dependent on the ensemble
        # state then run this ode system type instead
        if self.mode == "vary":
            cinds = np.asarray(self.ode_pop["cind"])
            ind_contact = []
            for ci in cinds:
                ind_contact.append(cont_mat[ci][cinds])
            ind_contact = np.asarray(ind_contact)
            totN = np.sum(Ns)

            def Lams_function(p, groups_Lams):
                contp = np.tensordot(ind_contact, p, axes=([1], [1])).swapaxes(0, 1)
                return groups_Lams + (contp * Ns / totN)

        # Define a function which takes the ode system forward
        # in time with variable parameter keywords
        def next_step_with_params(
            qpn, 
            t, 
            dt=timescale, 
            groups_Lams=groups_Lams, 
            groups_fs=groups_fs, 
            groups_mus=groups_mus,
        ):
            qpn_new = qpn
            q, p, n = (
                qpn[0],
                qpn[1 : self.nstat + 1],
                qpn[self.nstat + 1 : 2 * self.nstat + 1],
            )
            due_mask = (t >= groups_vts) * (groups_vts != -1.0)
            groups_vefs_deliv[due_mask] = groups_vefs[due_mask]
            gLs = Lams_function(p, groups_Lams)
            groups_Lams_v = gLs * (1.0 - groups_vefs_deliv)
            groups_sigsLams_v = np.minimum(
                groups_sigs * gLs, gLs * (1.0 - groups_vefs_deliv)
            )
            n_new = n + dt * (
                (groups_sigsLams_v * np.tensordot(np.ones(self.nstat), q, axes=0))
                + (
                    (
                        np.tensordot(
                            np.ones(self.nstat), np.sum(groups_fs * p, axis=0), axes=0,
                        )
                        - (groups_fs * p)
                    )
                    * groups_sigsLams_v
                )
            )
            F = (groups_Lams_v * np.exp(-n)) + (groups_sigsLams_v * (1.0 - np.exp(-n)))
            G = groups_mumaxs + (groups_mus - groups_mumaxs) * np.exp(
                np.tensordot(np.ones(self.nstat), np.sum(n, axis=0), axes=0)
                * (np.exp(-groups_epss) - 1.0)
            )
            q_new = q + dt * (np.sum(G * p, axis=0) - (np.sum(F, axis=0) * q))
            p_new = p + dt * (
                (F * q)
                + (
                    (
                        np.tensordot(
                            np.ones(self.nstat), np.sum(groups_fs * p, axis=0), axes=0,
                        )
                        - (groups_fs * p)
                    )
                    * F
                )
                - (G * p)
                - (
                    (np.tensordot(np.ones(self.nstat), np.sum(F, axis=0), axes=0) - F)
                    * groups_fs
                    * p
                )
            )
            (
                qpn_new[0],
                qpn_new[1 : self.nstat + 1],
                qpn_new[self.nstat + 1 : 2 * self.nstat + 1],
            ) = (q_new, p_new, n_new)
            return qpn_new

        # Define a function which runs the system over the specified
        # time period and computes the log-likelihood given the parameters
        def run_ode_compute_lnlike(
            qpn0, 
            t0, 
            tend, 
            dt=timescale,
            groups_Lams=groups_Lams, 
            groups_fs=groups_fs, 
            groups_mus=groups_mus,
        ):
            qpn = qpn0
            steps = int((tend - t0) / dt)
            t, past_t = t0, t0
            lnlike = 0.0
            for i in range(0, steps):
                qpn = next_step_with_params(
                    qpn,
                    t,
                    dt=dt,
                    groups_Lams=groups_Lams, 
                    groups_fs=groups_fs, 
                    groups_mus=groups_mus,
                )
                past_t = t
                t += dt
                rec = ((t >= data_times) * (data_times > past_t)) == True
                if np.any(rec):
                    data_qp = qpn[: self.nstat + 1][(data_Currs[rec], data_groups[rec])]
                    lnlike += np.sum(data_counts[rec] * np.log(data_qp))
            return lnlike

        # Run the system with consistent initial conditions and generate
        # output dictionary
        q0 = np.zeros(num_of_groups)
        p0 = np.zeros((self.nstat, num_of_groups))
        n0 = groups_npasts
        q0[Currs == 0] = 1.0
        p0[
            groups_Currs
            == np.tensordot(
                np.arange(1, self.nstat + 1, 1), np.ones(num_of_groups), axes=0
            )
        ] = 1.0
        qpn0 = np.zeros((2 * self.nstat + 1, num_of_groups))
        qpn0[0] = q0
        qpn0[1 : self.nstat + 1] = p0
        qpn0[self.nstat + 1 : 2 * self.nstat + 1] = n0
        lnlike = run_ode_compute_lnlike(qpn0, 0, np.max(data_times) + timescale)
        return lnlike