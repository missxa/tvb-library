# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)

"""
Models based on Wong-Wang's work.
"""

from .base import ModelNumbaDfun, LOG, numpy, basic, arrays
from numba import guvectorize, float64

@guvectorize([(float64[:],)*11], '(n),(m)' + ',()'*8 + '->(n)', nopython=True)
def _numba_dfun(S, c, a, b, d, g, ts, w, j, io, dx):
    "Gufunc for reduced Wong-Wang model equations."

    if S[0] < 0.0:
        dx[0] = 0.0 - S[0]
    elif S[0] > 1.0:
        dx[0] = 1.0 - S[0]
    else:
        x = w[0]*j[0]*S[0] + io[0] + j[0]*c[0]
        h = (a[0]*x - b[0]) / (1 - numpy.exp(-d[0]*(a[0]*x - b[0])))
        dx[0] = - (S[0] / ts[0]) + (1.0 - S[0]) * h * g[0]


class ModifiedWongWang(ModelNumbaDfun):
    r"""
    .. [WW_2006] Kong-Fatt Wong and Xiao-Jing Wang,  *A Recurrent Network
                Mechanism of Time Integration in Perceptual Decisions*.
                Journal of Neuroscience 26(4), 1314-1328, 2006.
    .. [DPA_2013] Deco Gustavo, Ponce Alvarez Adrian, Dante Mantini, Gian Luca
                  Romani, Patric Hagmann and Maurizio Corbetta. *Resting-State
                  Functional Connectivity Emerges from Structurally and
                  Dynamically Shaped Slow Linear Fluctuations*. The Journal of
                  Neuroscience 32(27), 11239-11252, 2013.
    .. automethod:: ModifiedWongWang.__init__
    Equations taken from [DPA_2013]_ , page 11242
    .. math::
                 x_k       &=   w\,J_N \, S_k + I_o + J_N \mathbf\Gamma(S_k, S_j, u_{kj}),\\
                 H(x_k)    &=  \dfrac{ax_k - b}{1 - \exp(-d(ax_k -b))},\\
                 \dot{S}_k &= -\dfrac{S_k}{\tau_s} + (1 - S_k) \, H(x_k) \, \gamma
    """
    _ui_name = "Modified Reduced Wong-Wang"
    ui_configurable_parameters = ['a_i', 'b_i', 'a_e', 'b_e', 'd_e', 'd_i', 'gamma_e', 'gamma_i', 'tau_e', 'tau_i', 'w', 'J_N', 'I_o', 'J_e', 'J_i']

    #Define traited attributes for this model, these represent possible kwargs.
    a_e = arrays.FloatArray(
        label=":math:`a`",
        default=numpy.array([310., ]),
        range=basic.Range(lo=0.0, hi=570., step=0.01),
        doc="[n/C]. Input gain parameter, chosen to fit numerical solutions.",
        order=1)

    a_i = arrays.FloatArray(
        label=":math:`a`",
        default=numpy.array([615., ]),
        range=basic.Range(lo=0.0, hi=870., step=0.01),
        doc="[n/C]. Input gain parameter, chosen to fit numerical solutions.",
        order=2)

    b_i = arrays.FloatArray(
        label=":math:`b`",
        default=numpy.array([177., ]),
        range=basic.Range(lo=0.0, hi=190.0, step=0.01),
        doc="[kHz]. Input shift parameter chosen to fit numerical solutions.",
        order=3)

    b_e = arrays.FloatArray(
        label=":math:`b`",
        default=numpy.array([125., ]),
        range=basic.Range(lo=0.0, hi=150.0, step=0.01),
        doc="[kHz]. Input shift parameter chosen to fit numerical solutions.",
        order=4)

    d_e = arrays.FloatArray(
        label=":math:`d`",
        default=numpy.array([0.16, ]),
        range=basic.Range(lo=0.0, hi=200.0, step=0.01),
        doc="""[ms]. Parameter chosen to fit numerical solutions.""",
        order=5)

    d_i = arrays.FloatArray(
        label=":math:`d`",
        default=numpy.array([0.087, ]),
        range=basic.Range(lo=0.0, hi=200.0, step=0.01),
        doc="""[ms]. Parameter chosen to fit numerical solutions.""",
        order=6)

    gamma_e = arrays.FloatArray(
        label=r":math:`\gamma`",
        default=numpy.array([0.000641, ]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Kinetic parameter""",
        order=7)

    gamma_i = arrays.FloatArray(
        label=r":math:`\gamma`",
        default=numpy.array([0.001, ]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Kinetic parameter""",
        order=7)

    tau_e = arrays.FloatArray(
        label=r":math:`\tau_S`",
        default=numpy.array([100., ]),
        range=basic.Range(lo=50.0, hi=150.0, step=1.0),
        doc="""Kinetic parameter. NMDA decay time constant.""",
        order=8)

    tau_i = arrays.FloatArray(
        label=r":math:`\tau_S`",
        default=numpy.array([10., ]),
        range=basic.Range(lo=0.0, hi=150.0, step=1.0),
        doc="""Kinetic parameter. NMDA decay time constant.""",
        order=9)

    w = arrays.FloatArray(
        label=r":math:`w`",
        default=numpy.array([1.4, ]),
        range=basic.Range(lo=0.0, hi=5.0, step=0.01),
        doc="""Excitatory recurrence""",
        order=10)

    J_N = arrays.FloatArray(
        label=r":math:`J_{N}`",
        default=numpy.array([0.2, ]),
        range=basic.Range(lo=0.2, hi=0.5, step=0.001),
        doc="""Excitatory recurrence""",
        order=11)

    J = arrays.FloatArray(
        label=r":math:`J_{N}`",
        default=numpy.array([1., ]),
        range=basic.Range(lo=0.2609, hi=1., step=0.001),
        doc="""Excitatory recurrence""",
        order=12)

    # J_i = arrays.FloatArray(
    #     label=r":math:`J_{N}`",
    #     default=numpy.array([0.7, ]),
    #     range=basic.Range(lo=0.2609, hi=1., step=0.001),
    #     doc="""Excitatory recurrence""",
    #     order=13)

    I_o = arrays.FloatArray(
        label=":math:`I_{o}`",
        default=numpy.array([0.382, ]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.01),
        doc="""[nA] Effective external input""",
        order=14)

    sigma_noise = arrays.FloatArray(
        label=r":math:`\sigma_{noise}`",
        default=numpy.array([0.000000001, ]),
        range=basic.Range(lo=0.0, hi=0.005),
        doc="""[nA] Noise amplitude. Take this value into account for stochatic
        integration schemes.""",
        order=-1)

    W_i = arrays.FloatArray(
        label=":math:`I_{o}`",
        default=numpy.array([0.7, ]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.01),
        doc="""[nA] Effective external input""",
        order=15)

    W_e = arrays.FloatArray(
        label=":math:`I_{o}`",
        default=numpy.array([1., ]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.01),
        doc="""[nA] Effective external input""",
        order=16)

    gcf = arrays.FloatArray(
        label=":math:`I_{o}`",
        default=numpy.array([0.11, ]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.01),
        doc="""[nA] Effective external input""",
        order=17)

    w_bg_e = arrays.FloatArray(
        label=":math:`I_{o}`",
        default=numpy.array([0.03, ]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.01),
        doc="""[nA] Effective external input""",
        order=18)

    w_bg_i = arrays.FloatArray(
        label=":math:`I_{o}`",
        default=numpy.array([0.15, ]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.01),
        doc="""[nA] Effective external input""",
        order=19)

    ext_input = arrays.FloatArray(
        label=":math:`I_{o}`",
        default=numpy.array([0., ]),
        range=basic.Range(lo=0.0, hi=10000.0, step=0.01),
        doc="""[nA] Effective external input""",
        order=20)

    state_variable_range = basic.Dict(
        label="State variable ranges [lo, hi]",
        default={"S_i": numpy.array([0.0, 1.0]), "S_e": numpy.array([0.0, 1.0])},
        doc="Population firing rate",
        order=21)

    variables_of_interest = basic.Enumerate(
        label="Variables watched by Monitors",
        options=["S_i", "S_e"],
        default=["S_i", "S_e"],
        select_multiple=True,
        doc="""default state variables to be monitored""",
        order=22)

    

    state_variables = ['S_i', 'S_e']
    _nvar = 2
    cvar = numpy.array([0,1], dtype=numpy.int32)

    def configure(self):
        """  """
        super(ModifiedWongWang, self).configure()
        self.update_derived_parameters()

    # def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0):
    #     r"""
    #     Equations taken from [DPA_2013]_ , page 11242
    #     .. math::
    #              x_k       &=   w\,J_N \, S_k + I_o + J_N \mathbf\Gamma(S_k, S_j, u_{kj}),\\
    #              H(x_k)    &=  \dfrac{ax_k - b}{1 - \exp(-d(ax_k -b))},\\
    #              \dot{S}_k &= -\dfrac{S_k}{\tau_s} + (1 - S_k) \, H(x_k) \, \gamma
    #     """
    #     S_e   = state_variables[0, :]
    #     S_i = state_variables[1, :]
    #     S_e[S_e<0] = 0.
    #     S_e[S_e>1] = 1.
    #     S_i[S_i<0] = 0.
    #     S_i[S_i>1] = 1.
    #     c_0 = coupling[0, :]


    #     # if applicable
    #     # lc_0 = local_coupling * S


    #     x_e = self.W_e * self.I_o + self.gcf * S_e - self.J * S_i + self.ext_input * self.w_bg_e
    #     x_i = self.W_i * self.I_o - S_i + self.ext_input * self.w_bg_i

    #     H_e = (self.a_e * x_e - self.b_e) / (1 - numpy.exp(-self.d_e * (self.a_e * x_e - self.b_e)))
    #     H_i = (self.a_i * x_i - self.b_i) / (1 - numpy.exp(-self.d_i * (self.a_e * x_i - self.b_i)))

    #     dS_e = - (S_e / self.tau_e) + (1 - S_e) * H_e * self.gamma_e
    #     dS_i = - (S_e / self.tau_i) + (1 - S_i) * H_i * self.gamma_i

    #     # x  = self.w * self.J_N * S + self.I_o + self.J_N * c_0 + self.J_N * lc_0
    #     # H = (self.a * x - self.b) / (1 - numpy.exp(-self.d * (self.a * x - self.b)))
    #     # dS = - (S / self.tau_s) + (1 - S) * H * self.gamma

    #     derivative_e = numpy.array([dS_e])
    #     derivative_e = numpy.array([dS_i])
    #     return (derivative_e, derivative_i)

    def dfun(self, state_variables, coupling, local_coupling=0.0):

        S_e = state_variables[0, :]
        S_i = state_variables[1, :]

        derivative = numpy.empty_like(state_variables)

        S_e[S_e<0] = 0.
        S_e[S_e>1] = 1.
        S_i[S_i<0] = 0.
        S_i[S_i>1] = 1.

        c_0 = coupling[0, :]
        # lc_0 = local_coupling * S_i

        x_e = self.W_e * self.I_o + self.gcf * c_0 - self.J * S_i + self.ext_input * self.w_bg_e
        x_i = self.W_i * self.I_o - S_i + self.ext_input * self.w_bg_i

        H_e = (self.a_e * x_e - self.b_e) / (1 - numpy.exp(-self.d_e * (self.a_e * x_e - self.b_e)))
        H_i = (self.a_i * x_i - self.b_i) / (1 - numpy.exp(-self.d_i * (self.a_e * x_i - self.b_i)))


        derivative[0] = - (S_e / self.tau_e) + (1 - S_e) * H_e * self.gamma_e
        derivative[1] = - (S_e / self.tau_i) + H_i * self.gamma_i


        return derivative