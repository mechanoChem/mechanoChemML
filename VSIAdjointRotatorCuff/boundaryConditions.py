Tendon20231012 = {
  "equations": {
    "Intact": {
      "tendon": lambda f: f.normal(0)<-0.8 and f.exterior() and f.midpoint()[0]<-15.,
      "enthesis": lambda f: f.normal(1)<-0.7 and f.exterior() and \
        f.midpoint()[0]-3.18e-4*f.midpoint()[2]**4+0.0033*f.midpoint()[2]**3-\
            0.0023*f.midpoint()[2]**2-0.3170*f.midpoint()[2]-0.7776 > 0.,
      "head": lambda f: f.normal(1)<-0.3 and f.exterior() and \
        f.midpoint()[0]-3.18e-4*f.midpoint()[2]**4+0.0033*f.midpoint()[2]**3-\
            0.0023*f.midpoint()[2]**2-0.3170*f.midpoint()[2]-0.7776 < 0. and \
                f.midpoint()[0]-0.0551*f.midpoint()[2]**2-\
                    0.2501*f.midpoint()[2]+13.8244 > 0.
    },
    "Torn": {
    "tendon": lambda f: f.normal(0)<-0.8 and f.exterior() and f.midpoint()[0]<-15.,
    "enthesis": lambda f: f.normal(1)<-0.7 and f.exterior() and \
      f.midpoint()[0]-3.18e-4*f.midpoint()[2]**4+0.0033*f.midpoint()[2]**3-\
        0.0023*f.midpoint()[2]**2-0.3170*f.midpoint()[2]-0.7776 > 0. \
          and (f.midpoint()[0]-0.0010*f.midpoint()[2]**4\
            -0.0139*f.midpoint()[2]**3 - 0.1176*f.midpoint()[2]**2 \
                - 0.3674*f.midpoint()[2] - 3.4498 < 0. or \
                    f.midpoint()[2]+0.0009*f.midpoint()[0]**3+\
                        0.1801*f.midpoint()[0]**2\
                        -3.2451*f.midpoint()[0] + 9.6708 > 0.),
    "head": lambda f: f.normal(1)<-0.5 and f.exterior() and \
      f.midpoint()[0]-3.18e-4*f.midpoint()[2]**4+0.0033*f.midpoint()[2]**3-\
        0.0023*f.midpoint()[2]**2-0.3170*f.midpoint()[2]-0.7776 < 0. and \
            f.midpoint()[0]-0.0551*f.midpoint()[2]**2-\
              0.2501*f.midpoint()[2]+13.8244 > 0.
    }
  }
}