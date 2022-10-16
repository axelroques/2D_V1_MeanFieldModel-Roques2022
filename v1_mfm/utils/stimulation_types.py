
import numpy as np


def stim_CENTER(t, X, Z, params, triple_gaussian):

    # Initialization
    t = params['t']
    X0 = X[int(len(X)/2.)]
    Z0 = Z[int(len(Z)/2.)]

    # Check if temporal boundary is ok, making it longer if stim not fully developed
    params['tstart'] = np.max([3.*params['Tau1'], params['tstart']])
    params['tstop'] = np.max(
        [params['tstart']+3.*params['Tau2'], params['tstop']])

    X1, t1, Z1 = np.meshgrid(X, t, Z)
    nu_e_aff = triple_gaussian(
        t1, X1, Z1, params['tstart'],
        params['Tau1'], params['Tau2'],
        X0, Z0, params['sX'], params['sZ'], params['amp']
    )

    return nu_e_aff


def stim_DOT(t, X, Z, params, triple_gaussian):

    # Initialization
    X0 = X[int(len(X)*2/5.)]
    Z0 = Z[int(len(Z)*4/5.)]

    # Check if temporal boundary is ok, making it longer if stim not fully developed
    params['tstart'] = np.max([3.*params['Tau1'], params['tstart']])
    params['tstop'] = np.max(
        [params['tstart']+3.*params['Tau2'], params['tstop']])

    X1, t1, Z1 = np.meshgrid(X, t, Z)
    nu_e_aff = triple_gaussian(
        t1, X1, Z1, params['tstart'],
        10e-3, 150e-3,
        X0, Z0, params['sX'], params['sZ'], params['amp'])

    return nu_e_aff


def stim_DOT_LEFT(t, X, Z, params, triple_gaussian):

    # Initialization
    t = params['t']
    X0 = X[int(len(X)/3.)]
    Z0 = Z[int(len(Z)/2.)]

    # Check if temporal boundary is ok, making it longer if stim not fully developed
    params['tstart'] = np.max([3.*params['Tau1'], params['tstart']])
    params['tstop'] = np.max(
        [params['tstart']+3.*params['Tau2'], params['tstop']])

    X1, t1, Z1 = np.meshgrid(X, t, Z)
    nu_e_aff = triple_gaussian(
        t1, X1, Z1, params['tstart'],
        params['Tau1'], params['Tau2'],
        X0, Z0, params['sX'], params['sZ'], params['amp'])

    return nu_e_aff


def stim_DOT_RIGHT(t, X, Z, params, triple_gaussian):

    # Initialization
    t = params['t']
    X0 = X[int(len(X)*2./3.)]
    Z0 = Z[int(len(Z)/2.)]

    # Check if temporal boundary is ok, making it longer if stim not fully developed
    params['tstart'] = np.max([3.*params['Tau1'], params['tstart']])
    params['tstop'] = np.max(
        [params['tstart']+3.*params['Tau2'], params['tstop']])

    X1, t1, Z1 = np.meshgrid(X, t, Z)
    nu_e_aff = triple_gaussian(
        t1, X1, Z1, params['tstart'] + 100e-3,
        params['Tau1'], params['Tau2'],
        X0, Z0, params['sX'], params['sZ'], params['amp'])

    return nu_e_aff


def stim_DOUBLE(t, X, Z, params, triple_gaussian):

    # Initialization
    X0_1 = X[int(len(X)/3.)]
    X0_2 = X[int(len(X)*2/3.)]
    Z0 = Z[int(len(Z)/2.)]
    print(X0_1, X0_2)

    # Check if temporal boundary is ok, making it longer if stim not fully developed
    params['tstart'] = np.max([3.*params['Tau1'], params['tstart']])
    params['tstop'] = np.max(
        [params['tstart']+3.*params['Tau2'], params['tstop']])

    # Calculation of triple gaussian with a meshgrid
    X1, t1, Z1 = np.meshgrid(X, t, Z)
    nu_e_aff = triple_gaussian(
        t1, X1, Z1, params['tstart'],
        params['Tau1'], params['Tau2'],
        X0_1, Z0, params['sX'], params['sZ'], params['amp']) + \
        triple_gaussian(
        t1, X1, Z1, params['tstart'] + 100e-3,
        params['Tau1'], params['Tau2'],
        X0_2, Z0, params['sX'], params['sZ'], params['amp'])

    return nu_e_aff


def stim_MOV_HORIZ_LINE(t, X, Z, params, triple_gaussian):

    # Check if temporal boundary is ok, making it longer if stim not fully developed
    params['tstart'] = np.max([3.*params['Tau1'], params['tstart']])
    params['tstop'] = np.max(
        [params['tstart']+3.*params['Tau2'], params['tstop']])

    # Complex stimuli generation
    X1, t1, Z1 = np.meshgrid(X, t, Z)
    # Draws horizontal line
    Z0 = Z[int(len(Z)/2.)]
    X0 = []
    for x in range(2, len(X)-3):
        X0.append(X[x])
    print('\t Visual stimuli:')
    print(X0)
    print([Z0])
    nu_e_aff = np.zeros((len(t), len(X), len(Z)))
    for pos in range(len(X0)):
        nu_e_aff += triple_gaussian(
            t1, X1, Z1, params['tstart'] + pos*10e-3,
            params['Tau1'], params['Tau2'],
            X0[pos], Z0, params['sX'], params['sZ'], params['amp'])

    return nu_e_aff


def stim_MOV_VERT_LINE(t, X, Z, params, triple_gaussian):

    # Check if temporal boundary is ok, making it longer if stim not fully developed
    params['tstart'] = np.max([3.*params['Tau1'], params['tstart']])
    params['tstop'] = np.max(
        [params['tstart']+3.*params['Tau2'], params['tstop']])

    # Complex stimuli generation
    X1, t1, Z1 = np.meshgrid(X, t, Z)
    X0 = Z[int(len(Z)*3/5.)]
    Z0 = []
    for z in range(len(Z)-3, 2, -3):
        Z0.append(Z[z])
    print('\t Visual stimuli:')
    print([X0])
    print(Z0)
    nu_e_aff = np.zeros((len(t), len(X), len(Z)))
    for pos in range(len(Z0)):
        nu_e_aff += triple_gaussian(
            t1, X1, Z1, params['tstart'] +
            100e-3 + pos*2.5e-3,
            10e-3, 50e-3,
            X0, Z0[pos], params['sX'], params['sZ'], params['amp'])

    return nu_e_aff


def stim_VERT_LINE(t, X, Z, params, triple_gaussian):

    # Check if temporal boundary is ok, making it longer if stim not fully developed
    params['tstart'] = np.max([3.*params['Tau1'], params['tstart']])
    params['tstop'] = np.max(
        [params['tstart']+3.*params['Tau2'], params['tstop']])

    # Complex stimuli generation
    X1, t1, Z1 = np.meshgrid(X, t, Z)
    X0 = Z[int(len(Z)*3/5.)]
    Z0 = []
    for z in range(2, len(Z)-3, 3):
        Z0.append(Z[z])
    print('\t Visual stimuli:')
    print([X0])
    print(Z0)
    nu_e_aff = np.zeros((len(t), len(X), len(Z)))
    for pos in range(len(Z0)):
        nu_e_aff += triple_gaussian(
            t1, X1, Z1, params['tstart'] + 100e-3,
            10e-3, 130e-3,
            X0, Z0[pos], params['sX'], params['sZ'], params['amp'])

    return nu_e_aff


def stim_LINE_MOTION(t, X, Z, params, triple_gaussian):

    # Check if temporal boundary is ok, making it longer if stim not fully developed
    params['tstart'] = np.max([3.*params['Tau1'], params['tstart']])
    params['tstop'] = np.max(
        [params['tstart']+3.*params['Tau2'], params['tstop']])

    # Complex stimuli generation
    X1, t1, Z1 = np.meshgrid(X, t, Z)

    # Dot
    X0_dot = X[int(len(X)*2/5.)]
    Z0_dot = Z[int(len(Z)*4/5.)]

    # Vertical Line
    X0_line = Z[int(len(Z)*3/5.)]
    Z0_line = []
    for z in range(2, len(Z)-3, 3):
        Z0_line.append(Z[z])
    print('\t Visual stimuli: line motion')
    nu_e_aff = np.zeros((len(t), len(X), len(Z)))
    nu_e_aff += triple_gaussian(
        t1, X1, Z1, params['tstart'],
        10e-3, 150e-3,
        X0_dot, Z0_dot, params['sX'], params['sZ'], params['amp'])
    for pos in range(len(Z0_line)):
        nu_e_aff += triple_gaussian(
            t1, X1, Z1, params['tstart'] + 100e-3,
            10e-3, 130e-3,
            X0_line, Z0_line[pos], params['sX'], params['sZ'], params['amp'])

    return nu_e_aff


def stim_DOUBLE_LINE(t, X, Z, params, triple_gaussian):

    # Check if temporal boundary is ok, making it longer if stim not fully developed
    params['tstart'] = np.max([3.*params['Tau1'], params['tstart']])
    params['tstop'] = np.max(
        [params['tstart']+3.*params['Tau2'], params['tstop']])

    # Complex stimuli generation
    X1, t1, Z1 = np.meshgrid(X, t, Z)
    # Draws horizontal lines
    Z01 = Z[int(len(Z)*2/3.)]
    Z02 = Z[int(len(Z)/3.)]
    X01 = []
    X02 = []
    for x in range(2, len(X)-3):
        X01.append(X[x])
        X02.append(X[-x-1])
    print('\t Visual stimuli:')
    print('X01', X01)
    print('X02', X01)
    print('Z01', [Z01])
    print('Z02', [Z02])
    nu_e_aff = np.zeros((len(t), len(X), len(Z)))
    for pos in range(len(X01)):
        nu_e_aff += triple_gaussian(
            t1, X1, Z1, params['tstart'] + pos*10e-3,
            params['Tau1'], params['Tau2'],
            X01[pos], Z01, params['sX'], params['sZ'], params['amp']) +\
            triple_gaussian(
            t1, X1, Z1, params['tstart'] + pos*10e-3,
            params['Tau1'], params['Tau2'],
            X02[pos], Z02, params['sX'], params['sZ'], params['amp'])

    return nu_e_aff


def stim_SMILEY(t, X, Z, params, triple_gaussian):

    # Check if temporal boundary is ok, making it longer if stim not fully developed
    params['tstart'] = np.max([3.*params['Tau1'], params['tstart']])
    params['tstop'] = np.max(
        [params['tstart']+3.*params['Tau2'], params['tstop']])

    # Complex stimuli generation
    X1, t1, Z1 = np.meshgrid(X, t, Z)
    radius = 10
    Nframes = 6
    x0 = X[int(len(X)/2.)]
    z0 = Z[int(len(Z)/2.5)]
    X0_smile = []
    Z0_smile = []
    for i in range(Nframes):
        time = 2*np.pi*float(i/(Nframes - 1.))/2
        X0_smile.append(x0 - radius*np.cos(time))
        Z0_smile.append(z0 - radius*np.sin(time))
    # Eye O
    radius = 1
    Nframes = 6
    x0 = X[int(len(X)/3)]
    z0 = Z[int(len(Z)*2/3)]
    X0_O = []
    Z0_O = []
    for i in range(Nframes):
        time = 2*np.pi*float(i/(Nframes - 1.))
        X0_O.append(x0 - radius*np.cos(time))
        Z0_O.append(z0 - radius*np.sin(time))
    # Eye -
    Z0_i = []
    X0_i = []
    for x in range(16, 22):
        X0_i.append(X[x])
        Z0_i.append(Z[int(len(Z)*2/3.)])
    print('\t Visual stimuli: ;)')
    nu_e_aff = np.zeros((len(t), len(X), len(Z)))
    for pos in range(len(X0_i)):
        nu_e_aff += triple_gaussian(
            t1, X1, Z1, params['tstart'] + pos*50e-3,
            params['Tau1'], params['Tau2'],
            X0_i[pos], Z0_i[pos], params['sX'], params['sZ'], params['amp']) +\
            triple_gaussian(
            t1, X1, Z1, params['tstart'] + pos*50e-3,
            params['Tau1'], params['Tau2'],
            X0_O[pos], Z0_O[pos], params['sX'], params['sZ'], params['amp']) +\
            triple_gaussian(
            t1, X1, Z1, params['tstart'] + pos*50e-3,
            params['Tau1'], params['Tau2'],
            X0_smile[pos], Z0_smile[pos], params['sX'], params['sZ'], params['amp'])

    return nu_e_aff


def stim_RECTANGLE(t, X, Z, params, triple_gaussian):

    # Check if temporal boundary is ok, making it longer if stim not fully developed
    params['tstart'] = np.max([3.*params['Tau1'], params['tstart']])
    params['tstop'] = np.max(
        [params['tstart']+3.*params['Tau2'], params['tstop']])

    # Complex stimuli generation
    X1, t1, Z1 = np.meshgrid(X, t, Z)
    # Draws the rectangle
    Z0 = []
    X0 = []
    for x in range(2, len(X)-3):
        X0.append(X[x])
        Z0.append(Z[int(len(Z)*2/3.)])
    for z in range(int(len(Z)*2/3.), int(len(Z)*1/3.)-1, -1):
        X0.append(X[len(X)-3])
        Z0.append(Z[z])
    for x in range(len(X)-3, 1, -1):
        X0.append(X[x])
        Z0.append(Z[int(len(Z)*1/3.)])
    for z in range(int(len(Z)*1/3.), int(len(Z)*2/3.)+1):
        X0.append(X[2])
        Z0.append(Z[z])
    print('\t Visual stimuli:')
    print(X0)
    print(Z0)
    nu_e_aff = np.zeros((len(t), len(X), len(Z)))
    for pos in range(len(X0)):
        nu_e_aff += triple_gaussian(
            t1, X1, Z1, params['tstart'] + pos*10e-3,
            params['Tau1'], params['Tau2'],
            X0[pos], Z0[pos], params['sX'], params['sZ'], params['amp'])

    return nu_e_aff


def stim_CIRCLE(t, X, Z, params, triple_gaussian):

    # Check if temporal boundary is ok, making it longer if stim not fully developed
    params['tstart'] = np.max([3.*params['Tau1'], params['tstart']])
    params['tstop'] = np.max(
        [params['tstart']+3.*params['Tau2'], params['tstop']])

    # Complex stimuli generation
    X1, t1, Z1 = np.meshgrid(X, t, Z)
    # Draws the circle
    radius = 10
    t_increment = 100
    x0 = X[int(len(X)/2.)]
    z0 = Z[int(len(Z)/2.)]
    X0 = []
    Z0 = []
    for i in range(t_increment):
        time = 2.*np.pi*float(i/(t_increment - 1.))
        X0.append(x0 - radius*np.cos(time))
        Z0.append(z0 - -radius*np.sin(time))
    print('\t Visual stimuli:')
    print(X0)
    print(Z0)
    nu_e_aff = np.zeros((len(t), len(X), len(Z)))
    for pos in range(len(X0)):
        nu_e_aff += triple_gaussian(
            t1, X1, Z1, params['tstart'] + pos*10e-3,
            params['Tau1'], params['Tau2'],
            X0[pos], Z0[pos], params['sX'], params['sZ'], params['amp'])

    return nu_e_aff


def stim_RANDOM(t, X, Z, params, triple_gaussian):
    # Check if temporal boundary is ok, making it longer if stim not fully developed
    params['tstart'] = np.max([3.*params['Tau1'], params['tstart']])
    params['tstop'] = np.max(
        [params['tstart']+3.*params['Tau2'], params['tstop']])

    # Complex stimuli generation
    X1, t1, Z1 = np.meshgrid(X, t, Z)
    # Draws random dots
    np.random.seed(19)
    rand_events = np.random.randint(1, len(t)//2)
    print(rand_events, 'random events')
    Z0 = np.random.randint(int(Z[len(Z)-1]), size=rand_events)
    X0 = np.random.randint(int(Z[len(X)-1]), size=rand_events)
    print('\t Visual stimuli:')
    print(X0)
    print(Z0)
    nu_e_aff = np.zeros((len(t), len(X), len(Z)))
    for pos in range(len(X0)):
        nu_e_aff += triple_gaussian(
            t1, X1, Z1, params['tstart'] + pos*10e-3,
            params['Tau1'], params['Tau2'],
            X0[pos], Z0[pos], params['sX'], params['sZ'], params['amp'])

    return nu_e_aff
