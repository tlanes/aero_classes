import numpy as np
import astropy.coordinates as coord
import astropy.units as u
import datetime


class OrbitClass:

    def __init__(self, epoch=datetime.datetime(2000, 1, 1, 12, 0, 0)):
        self.ephem_type_set = 'kepler'
        # Kepler elements
        self.a = 6378e3 + 500e3
        self.e = 1e-4
        self.i = 45.0 * np.pi / 180.0
        self.w = 90.0 * np.pi / 180.0
        self.O = 10.0 * np.pi / 180.0
        self.f = 0
        self.m = 0
        self.E = 0

        self.eci_r = np.array([0, 0, 0])
        self.eci_v = np.array([0, 0, 0])
        # Cartesian elements
        self.kep2car()

        if epoch is not None:
            x, y, z = self.eci_r[0] * u.m, self.eci_r[1] * u.m, self.eci_r[2] * u.m
            v_x, v_y, v_z = self.eci_v[0] * 1e-3 * u.km / u.s, self.eci_v[1] * 1e-3 * u.km / u.s, self.eci_v[
                2] * 1e-3 * u.km / u.s
            gcrs = coord.GCRS(x=x, y=y, z=z,
                              v_x=v_x, v_y=v_y, v_z=v_z,
                              representation_type='cartesian', differential_type='cartesian', obstime=epoch)
            itrs = gcrs.transform_to(coord.ITRS(obstime=epoch))
            e_r = itrs.cartesian.xyz.value
            e_vx, e_vy, e_vz = itrs.cartesian.differentials['s'].d_x.value, itrs.cartesian.differentials['s'].d_y.value, \
                               itrs.cartesian.differentials['s'].d_z.value
            self.ecf_r = e_r
            self.ecf_v = np.array([e_vx, e_vy, e_vz]) * 1e3
        else:
            self.ecf_r = np.array([0, 0, 0])
            self.ecf_v = np.array([0, 0, 0])

        self.epoch = epoch

    def kep2car(self, mu=3.986004415e14):
        p = self.a * (1 - self.e ** 2)
        rmag = p / (1 + self.e * np.cos(self.f))
        P_r = rmag * np.array([np.cos(self.f), np.sin(self.f), 0])
        P_v = np.sqrt(mu / p) * np.array([-np.sin(self.f), (self.e + np.cos(self.f)), 0])
        N_P = np.array([[np.cos(self.O) * np.cos(self.w) - np.sin(self.O) * np.sin(self.w) * np.cos(self.i),
                         -np.cos(self.O) * np.sin(self.w) - np.sin(self.O) * np.cos(self.w) * np.cos(self.i),
                         np.sin(self.O) * np.sin(self.i)],
                        [np.sin(self.O) * np.cos(self.w) + np.cos(self.O) * np.sin(self.w) * np.cos(self.i),
                         -np.sin(self.O) * np.sin(self.w) + np.cos(self.O) * np.cos(self.w) * np.cos(self.i),
                         -np.cos(self.O) * np.sin(self.i)],
                        [np.sin(self.w) * np.sin(self.i), np.cos(self.w) * np.sin(self.i), np.cos(self.i)]])
        self.eci_r = N_P @ P_r
        self.eci_v = N_P @ P_v

    def car2kep(self, mu=3.986004415e14):
        h = np.cross(self.eci_r, self.eci_v)
        hmag = np.linalg.norm(h)
        evec = np.cross(self.eci_v, h) / mu - self.eci_r / np.linalg.norm(self.eci_r)
        self.e = np.linalg.norm(evec)
        nhat = np.cross(np.array([0, 0, 1]), h)
        if np.dot(self.eci_r, self.eci_v) >= 0:
            self.f = np.arccos(np.dot(evec, self.eci_r)/self.e/np.linalg.norm(self.eci_r))
        else:
            self.f = 2*np.pi - np.arccos(np.dot(evec, self.eci_r)/self.e/np.linalg.norm(self.eci_r))

        self.i = np.arccos(h[-1]/hmag)
        self.E = 2 * np.arctan2(np.tan(self.f), np.sqrt((1+self.e)/(1-self.e)))
        if nhat[1] >= 0:
            self.O = np.arccos(nhat[0]/np.linalg.norm(nhat))
        else:
            self.O = 2*np.pi - np.arccos(nhat[0]/np.linalg.norm(nhat))

        if evec[2] >= 0:
            self.w = np.arccos(np.dot(nhat, evec)/np.linalg.norm(nhat)/self.e)
        else:
            self.w = 2*np.pi - np.arccos(np.dot(nhat, evec) / np.linalg.norm(nhat) / self.e)

        self.a = 1/(2/np.linalg.norm(self.eci_r) - np.linalg.norm(self.eci_v)**2 / mu)

    def set_kep(self, kep, epoch=None):
        self.ephem_type_set = 'kepler'
        self.a, self.e, self.i, self.w, self.O, self.f = kep
        self.eci_r = np.array([0, 0, 0])
        self.eci_v = np.array([0, 0, 0])
        # Cartesian elements
        self.kep2car()

        if epoch is not None:
            x, y, z = self.eci_r[0] * u.m, self.eci_r[1] * u.m, self.eci_r[2] * u.m
            v_x, v_y, v_z = self.eci_v[0] * 1e-3 * u.km / u.s, self.eci_v[1] * 1e-3 * u.km / u.s, self.eci_v[
                2] * 1e-3 * u.km / u.s
            gcrs = coord.GCRS(x=x, y=y, z=z,
                              v_x=v_x, v_y=v_y, v_z=v_z,
                              representation_type='cartesian', differential_type='cartesian', obstime=epoch)
            itrs = gcrs.transform_to(coord.ITRS(obstime=epoch))
            e_r = itrs.cartesian.xyz.value
            e_vx, e_vy, e_vz = itrs.cartesian.differentials['s'].d_x.value, itrs.cartesian.differentials['s'].d_y.value, \
                               itrs.cartesian.differentials['s'].d_z.value
            self.ecf_r = e_r
            self.ecf_v = np.array([e_vx, e_vy, e_vz]) * 1e3
        else:
            self.ecf_r = np.array([0, 0, 0])
            self.ecf_v = np.array([0, 0, 0])

        self.epoch = epoch

    def set_eci_car(self, r, v, epoch=None):
        self.ephem_type_set = 'cartesian'
        self.eci_r, self.eci_v = r, v

        # Cartesian elements
        self.car2kep()

        if epoch is not None:
            x, y, z = self.eci_r[0] * u.m, self.eci_r[1] * u.m, self.eci_r[2] * u.m
            v_x, v_y, v_z = self.eci_v[0] * 1e-3 * u.km / u.s, self.eci_v[1] * 1e-3 * u.km / u.s, self.eci_v[
                2] * 1e-3 * u.km / u.s
            gcrs = coord.GCRS(x=x, y=y, z=z,
                              v_x=v_x, v_y=v_y, v_z=v_z,
                              representation_type='cartesian', differential_type='cartesian', obstime=epoch)
            itrs = gcrs.transform_to(coord.ITRS(obstime=epoch))
            e_r = itrs.cartesian.xyz.value
            e_vx, e_vy, e_vz = itrs.cartesian.differentials['s'].d_x.value, itrs.cartesian.differentials['s'].d_y.value, \
                               itrs.cartesian.differentials['s'].d_z.value
            self.ecf_r = e_r
            self.ecf_v = np.array([e_vx, e_vy, e_vz]) * 1e3
        else:
            self.ecf_r = np.array([0, 0, 0])
            self.ecf_v = np.array([0, 0, 0])

        self.epoch = epoch

    def set_ecf_car(self, r, v, epoch=None):
        self.ephem_type_set = 'cartesian'
        self.ecf_r, self.ecf_v = r, v

        if epoch is not None:
            x, y, z = self.ecf_r[0] * u.m, self.ecf_r[1] * u.m, self.ecf_r[2] * u.m
            v_x, v_y, v_z = self.ecf_v[0] * 1e-3 * u.km / u.s, self.ecf_v[1] * 1e-3 * u.km / u.s, self.ecf_v[
                2] * 1e-3 * u.km / u.s
            itrs = coord.ITRS(x=x, y=y, z=z,
                              v_x=v_x, v_y=v_y, v_z=v_z,
                              representation_type='cartesian', differential_type='cartesian', obstime=epoch)
            gcrs = itrs.transform_to(coord.GCRS(obstime=epoch))
            e_r = gcrs.cartesian.xyz.value
            e_vx, e_vy, e_vz = gcrs.cartesian.differentials['s'].d_x.value, gcrs.cartesian.differentials['s'].d_y.value, \
                               gcrs.cartesian.differentials['s'].d_z.value
            self.eci_r = e_r
            self.eci_v = np.array([e_vx, e_vy, e_vz]) * 1e3

            self.car2kep()

        else:
            self.eci_r = np.array([0, 0, 0])
            self.eci_v = np.array([0, 0, 0])

        self.epoch = epoch

    def kepler_eq(self, n=1000, tol=1e-4):
        En = self.f
        Enp1 = self.f
        for idx in range(n):
            Enp1 = En - (En - self.e*np.sin(En) - self.m)/(1 - self.e*np.cos(En))
            if abs(Enp1-En) < tol:
                break
            En = Enp1
        self.E = Enp1


class OrbitProp(OrbitClass):
    def __init__(self, start_date=None, stop_date=None, dt=1, epoch=None):
        OrbitClass.__init__(self, epoch)

        self.ck2 = 5.413080e-4 # 1/2 * J2 * a_E^2
        self.ck4 = 0.62098875e-6 # -3/8 * J4 * a_E^4
        self.e6a = 1.0e-6
        self.qoms2t = 1.88027916e-9 # (q_0 - s)^4 * (er)^4
        self.s =  1.01222928
        self.tothird = 2/3
        self.xj3 = -0.253881e-5 # J3
        self.xke = 0.743669161E-1
        self.xkmper = 6378.135
        self.xmnpda = 1440.0
        self.ae = 1.0
        self.de2ra = np.pi/180.
        self.j2 = 1.08262668e-3
        self.j3 = -0.253881e-5


        self.propagate_sgp(start_date, dt)

        a=1

    def propagate_sgp(self, start, dt, mu=3.986004415e9):
        ke = np.sqrt(mu)
        n0 = np.sqrt(mu/self.a**3)
        a1 = (ke/n0)**(3/2)
        del1 = (3/2)*self.ck2/a1**2*(3*np.cos(self.i)**2-1)/(1-self.e**2)**(3/2)
        a0 = a1 * (1 - (1/3)*del1 - del1**2 - 134/81*del1**3)
        p0 = a0 * (1 - self.e**2)
        q0 = a0 * (1 - self.e)
        L0 = self.m + self.w + self.O
        dOdt = -3*self.ck2*n0*np.cos(self.i)/p0**2
        dwdt = (3/2)*self.ck2*n0*(5*np.cos(self.i)**2-1)/p0**2

        # assume ndot, nddot = 0
        a = a0 * n0**(3/2)
        if a > q0:
            e = 1 - q0/a
        else:
            e = 10**(-6)
        p  = a * (1 - e**2)
        Os0 = self.O + dOdt*dt
        ws0 = self.w + dwdt*dt
        Ls = L0 + (n0 + dwdt + dOdt)*dt
        ayNSL = e * np.sin(ws0) - (1/2)*self.j3*self.ae/self.j2/p*np.sin(self.i)
        axNSL = e * np.cos(ws0)
        L = Ls - (1/4)*self.j3*self.ae/self.j2/p*axNSL*np.sin(self.i)*(3+5*np.cos(self.i))/(1+np.cos(self.i))

        U = L - Os0
        Epwi = U
        for ii in range(100):
            dEpw = (U - ayNSL*np.cos(Epwi) + axNSL*np.sin(Epwi) - Epwi)/(-ayNSL*np.sin(Epwi) - axNSL*np.cos(Epwi) + 1)
            Epwip1 = Epwi + dEpw
            if abs(Epwip1 - Epwi) < 1e-4:
                Epwi = Epwip1
                break
            Epwi = Epwip1
        Epw = Epwi
        ecosE = axNSL*np.cos(Epw) + ayNSL*np.sin(Epw)
        esinE = axNSL*np.sin(Epw) - ayNSL*np.cos(Epw)
        eL2 = axNSL**2 + ayNSL**2
        pL = a * (1 - eL2)
        r = a * (1 - ecosE)
        rdot = ke * np.sqrt(a)/r * esinE
        rvdot = ke * np.sqrt(pL)/r
        sinu = (a/r)*(np.sin(Epw) - ayNSL - axNSL*esinE/(1+np.sqrt(1-eL2)))
        cosu = (a/r)*(np.cos(Epw) - axNSL + ayNSL*esinE/(1+np.sqrt(1-eL2)))
        u = np.arctan2(sinu, cosu)
        rk = r + (1/2)*self.ck2/pL*np.sin(self.i)**2 * np.cos(2*u)
        uk = u - (1/4)*self.ck2/pL**2 * (7*np.cos(self.i)**2 - 1)*np.sin(2*u)
        Ok = Os0 + (3/2)*self.ck2/pL**2 * np.cos(self.i) * np.sin(2*u)
        ik = self.i + (3/2)*self.ck2/pL**2 * np.sin(self.i)*np.cos(self.i)*np.cos(2*u)

        M = np.array([-np.sin(Ok)*np.cos(ik), np.cos(Ok)*np.cos(ik), np.sin(ik)])
        N = np.array([np.cos(Ok), np.sin(Ok), 0])
        U = M * np.sin(uk) + N * np.cos(uk)
        V = M * np.cos(uk) - N * np.sin(uk)

        eci_r = rk * U * 1e3
        eci_v = rdot * U + (rvdot) * V * 1e3
        self.set_eci_car(eci_r, eci_v, start+datetime.timedelta(seconds=dt))






if __name__ == "__main__":
    oc = OrbitClass()
    ob = OrbitClass()
    ob.car2kep()

    testepoch = datetime.datetime(2024, 4, 3, 14, 0, 0)
    oset_kep = OrbitClass(epoch=testepoch)
    oset_kep.set_kep([6378e3+300e3, 0.4, 30*np.pi/180, 5*np.pi/180, 20*np.pi/180, 90*np.pi/180])

    oset_careci = OrbitClass(epoch=testepoch)
    oset_careci.set_eci_car([-2114624.21954315,  4380428.81981792,  2794087.0414198], [-8944.54246445,  -836.96500241,  1312.15773182], epoch=testepoch)

    oset_carecf = OrbitClass(epoch=testepoch)
    oset_carecf.set_ecf_car([1361354.42602153, 4672609.93765161, 2789291.15147755], [-6861.01525833,  5276.1266007,  1291.13639693], epoch=testepoch)

    dt = datetime.timedelta(hours=1).seconds
    bb_class = OrbitProp(start_date=testepoch, stop_date=testepoch+datetime.timedelta(hours=1), dt=dt, epoch=testepoch)


    a = 1