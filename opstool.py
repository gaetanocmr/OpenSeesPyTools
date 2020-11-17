# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 17:48:35 2020

@author: gaeta
"""
             
                
def import_dxf(name):
    """
    Introduction
    ------------
    This function will import nodes and element connectivity in 2d plane, 
    starting for a .dxf CAD file.
    It will give as output connectivity matrix and node matrix
    
    Information
    -----------
        Author: Gaetano Camarda, Ph.D. Student in Structural Engineer
        Affiliation: University of Palermo
        e-mail: gaetano.camarda@unipa.it
        
                                                                     V1.0 2020
    Package needed
    --------------
    - ezdxf
        
    Input:
    ------
        name, file name, for example 'draw.dxf'
        NOTE:
        ----
            file name will contain patch folder if .dxf file is in another 
            folder.
    """
    if isinstance(name,str):
        import ezdxf
        doc = ezdxf.readfile(name)
        msp = doc.modelspace()
        lines = msp.query('LINE')
        type(lines)
        def save_entity(e):
            p1 = e.dxf.start
            p2 = e.dxf.end
            return p1, p2
        
        el = [] # el stands for element
        for e in msp.query('LINE'):
            el.append(save_entity(e))
            
        # Define Nodes
        def node_extract(el, length): 
            nodes = []
            for i in range(length):
                nodes.append(el[i][0])
                nodes.append(el[i][1])
            return nodes
        
        def removeDuplicates(nodes): 
            return [t for t in (set(tuple(i) for i in nodes))]
        
        el_new = []
        for i, value in enumerate(el): # Round nodes values
            tmp = [[round(ezdxf.math.xround(value[0][0], 0.001),2), \
                    round(ezdxf.math.xround(value[0][1], 0.001),2), \
                    round(ezdxf.math.xround(value[0][2], 0.001),2)], \
                   [round(ezdxf.math.xround(value[1][0], 0.001),2), \
                    round(ezdxf.math.xround(value[1][1], 0.001),2), \
                    round(ezdxf.math.xround(value[1][2], 0.001),2)]]
            el_new.append(tmp)
        
        def connectivity(el_new, nodes):
            c = []
            d = []
            for i in range(len(el_new)):
                for j in range(len(nodes)):
                    if el_new[i][1] == list(nodes[j]):
                        c.append(j)
                    elif el_new[i][0] == list(nodes[j]):
                        d.append(j)
            return c, d
        nodes = removeDuplicates(node_extract(el_new, len(el_new))) # Define nodelist for Openseespy <----
        connect = connectivity(el_new, nodes) # Define connectivity for each element <----
        return nodes, connect
    else:
        raise RuntimeError('Inserire un percorso di file dxf valido')
        
def select_node(nodes, coordinate = 0, column = 1):
    """
    Introduction
    ------------
    This function will return the index of selected value.
    For example in a node matrix |node number | xcoord | ycoord
    for xcoord_signed it will find all node with that coordinate and return it
    in a list of node value.
    
    Information
    -----------
        Author: Gaetano Camarda, Ph.D. Student in Structural Engineer
        Affiliation: University of Palermo
        e-mail: gaetano.camarda@unipa.it
        
                                                                     V1.0 2020
    Package needed
    --------------
    - numpy
        
    Input:
    ------
        nodes, matrix.array with nodes coordinate
        coordinate, the fixed value to search for
        column, the column containing the value to look for
    """
    import numpy as np 
    arr = np.array(nodes)
    ind = np.where(arr[:,column] == coordinate)
    ind = np.asarray(ind)
    ind = ind.tolist()
    return ind

#### DISMISSED FUNCTION ###
# def concrete_section(sectag, H, B, cover, nbary, fi, matcore,
#                      matcover, matsteel, nbarz = 0, numsubdivy = 10, numsubdivz = 2):
#     ######## dismissed ###############
#     import numpy as np
#     import openseespy.opensees as ops
#     ymax = B / 2
#     zmax = H / 2
#     ymin = ymax - cover
#     zmin = zmax - cover
#     Abar = np.pi * np.power(fi/2,2)
#     point_coordinate = {
#                         'A' : [ymax, zmax],
#                         'B' : [-ymax, zmax],
#                         'C' : [-ymax, -zmax],
#                         'D' : [ymax, -zmax],
#                         'E' : [-ymin, -zmin],
#                         'F' : [ymin, -zmin],
#                         'G' : [ymin, zmin],
#                         'H' : [-ymin, zmin],
#                         'E1' : [-ymin, -zmax],
#                         'B1' : [-ymin, zmax],
#                         'A1' : [ymin, zmax],
#                         'F1' : [ymin, -zmax]
#                         }
    
#     ops.section('Fiber', sectag)
#     ops.patch('rect',matcore, numsubdivy, numsubdivz, point_coordinate['C'][0],  point_coordinate['C'][1],
#               point_coordinate['A'][0], point_coordinate['A'][1]) #core 1
#     ops.patch('rect', matcover, numsubdivy, numsubdivz, point_coordinate['E1'][0], point_coordinate['E1'][1],
#               point_coordinate['F'][0], point_coordinate['F'][1]) #cover 2
#     ops.patch('rect', matcover,numsubdivz, numsubdivy, point_coordinate['C'][0], point_coordinate['C'][1],
#               point_coordinate['B1'][0], point_coordinate['B1'][1]) # cover 3
#     ops.patch('rect', matcover,numsubdivy, numsubdivz, point_coordinate['H'][0], point_coordinate['H'][1],
#               point_coordinate['A1'][0], point_coordinate['A1'][1]) # cover 4
#     ops.patch('rect', matcover,numsubdivz, numsubdivy, point_coordinate['F1'][0], point_coordinate['F1'][1],
#               point_coordinate['A'][0], point_coordinate['A'][1]) # cover 3
    
#     # Define Layer 
#     if nbary > 0:
#         # print('nbary')
#         ops.layer('straight', matsteel, nbary, Abar, point_coordinate['H'][0], point_coordinate['H'][1],
#                   point_coordinate['G'][0], point_coordinate['G'][1])
#         ops.layer('straight', matsteel, nbary, Abar, point_coordinate['E'][0], point_coordinate['E'][1],
#                   point_coordinate['F'][0], point_coordinate['F'][1])
#         fib_sec = [['section', 'Fiber', sectag],
#            ['patch','rect',matcore,numsubdivy, numsubdivz, point_coordinate['C'][0],  point_coordinate['C'][1], point_coordinate['A'][0], point_coordinate['A'][1]],
#            ['patch', 'rect', matcover, numsubdivy, numsubdivz, point_coordinate['E1'][0], point_coordinate['E1'][1],
#           point_coordinate['F'][0], point_coordinate['F'][1]],
#            ['patch', 'rect', matcover,numsubdivz, numsubdivy, point_coordinate['C'][0], point_coordinate['C'][1],
#           point_coordinate['B1'][0], point_coordinate['B1'][1]],
#            ['patch', 'rect', matcover,numsubdivy, numsubdivz, point_coordinate['H'][0], point_coordinate['H'][1],
#           point_coordinate['A1'][0], point_coordinate['A1'][1]],
#            ['patch', 'rect', matcover,numsubdivz, numsubdivy, point_coordinate['F1'][0], point_coordinate['F1'][1],
#           point_coordinate['A'][0], point_coordinate['A'][1]],
#            ['layer','straight', matsteel, nbary, Abar, point_coordinate['H'][0], point_coordinate['H'][1],
#           point_coordinate['G'][0], point_coordinate['G'][1]],
#            ['layer', 'straight', matsteel, nbary, Abar, point_coordinate['E'][0], point_coordinate['E'][1],
#           point_coordinate['F'][0], point_coordinate['F'][1]]
#            ]
#     if nbarz > 0:
#         # print('nbarz')
#         ops.layer('straight', matsteel, nbarz, Abar, point_coordinate['E'][0], point_coordinate['E'][1],
#                   point_coordinate['H'][0], point_coordinate['H'][1])
#         ops.layer('straight', matsteel, nbarz, Abar, point_coordinate['F'][0], point_coordinate['F'][1],
#                   point_coordinate['G'][0], point_coordinate['G'][1])
#         fib_sec = [['section', 'Fiber', sectag],
#                  ['patch','rect',matcore,numsubdivy, numsubdivz, point_coordinate['C'][0],  point_coordinate['C'][1], point_coordinate['A'][0], point_coordinate['A'][1]],
#                  ['patch', 'rect', matcover, numsubdivy, numsubdivz, point_coordinate['E1'][0], point_coordinate['E1'][1],
#                 point_coordinate['F'][0], point_coordinate['F'][1]],
#                  ['patch', 'rect', matcover,numsubdivz, numsubdivy, point_coordinate['C'][0], point_coordinate['C'][1],
#                 point_coordinate['B1'][0], point_coordinate['B1'][1]],
#                  ['patch', 'rect', matcover,numsubdivy, numsubdivz, point_coordinate['H'][0], point_coordinate['H'][1],
#                 point_coordinate['A1'][0], point_coordinate['A1'][1]],
#                  ['patch', 'rect', matcover,numsubdivz, numsubdivy, point_coordinate['F1'][0], point_coordinate['F1'][1],
#                 point_coordinate['A'][0], point_coordinate['A'][1]],
#                  ['layer','straight', matsteel, nbary, Abar, point_coordinate['H'][0], point_coordinate['H'][1],
#                 point_coordinate['G'][0], point_coordinate['G'][1]],
#                  ['layer', 'straight', matsteel, nbary, Abar, point_coordinate['E'][0], point_coordinate['E'][1],
#                 point_coordinate['F'][0], point_coordinate['F'][1]],
#                  ['layer', 'straight', matsteel, nbarz, Abar, point_coordinate['E'][0], point_coordinate['E'][1],
#                     point_coordinate['H'][0], point_coordinate['H'][1]],
#                  ['layer', 'straight', matsteel, nbarz, Abar, point_coordinate['F'][0], point_coordinate['F'][1],
#                     point_coordinate['G'][0], point_coordinate['G'][1]]
#                  ]       
#     return fib_sec
    
def bilinear_construction(push):
    """
    Introduction
    ------------
    Given a PushOver Curve, or a general curve this function will generate a 
    bilinear rappresentation as prescribed in Italian NTC2018 C7.3.4.2 Method A.
    The function will first increase number of numerical point if length given
    is less then 300, at this point the bilinear algorithm will start.
    Output will be 3 point that define the bilinear [A, B, C] and the matrix 
    nrow * 2 column, with the increased numerical point
    
    Information
    -----------
        Author: Gaetano Camarda, Ph.D. Student in Structural Engineer
        Affiliation: University of Palermo
        e-mail: gaetano.camarda@unipa.it
        
                                                                     V1.0 2020
    Package needed
    --------------
    - numpy
        
    Input:
    ------
        push matrix, dimension (n x 2), first column with displacement and 
        second column with forces.
        NOTE
        ----
        this function could be used with generic curve as well.
    """
    import numpy as np
    if len(push) < 300:
        for it in range(4):
            FS_temp = []
            for i in range(len(push)-1):
                x = (push[i+1,0] + push[i,0])/2
                y = (push[i+1,1] + push[i,1])/2
                FS_temp.append([push[i,0], push[i,1]])
                FS_temp.append([x, y])
            push = np.array(FS_temp)
            
    Fbu_star = 0.6 * np.max(push[:,1])
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    ind = find_nearest(push[:,1], Fbu_star)
    d_star = push[ind,0]
    K_star = Fbu_star / d_star
    tol = 1
    func = lambda x : K_star * x
    
    for j in range(1,len(push)):

        A_r1 = np.trapz(func(push[:j,0]), push[0:j,0])
        A_C1 = np.trapz(push[0:j,1], push[0:j,0])
        A1 = A_C1 - A_r1
        
        A_C2 = np.trapz(push[j:,1], push[j:,0])
        val = func(push[j,0])
        r_2 = np.ones(len(push[j:,0])) * val
        a_pos = find_nearest(push[:,1], val)
        A_r2 = np.trapz(r_2, push[j:,0])
        A2 = A_C2 - A_r2
        d_y = push[j,0]
    
        perc_A1 = (A1 / (A1+A2)) * 100;
        perc_A2 = (A2 / (A1+A2)) * 100;
        if A2 < A1:
            if perc_A1 - perc_A2 < tol:
                break
        if A1 < A2:
            if perc_A2 - perc_A1 < tol:
                break
    
    d_y = push[j,0]
    A =[d_star, Fbu_star]
    B = [d_y, push[a_pos,1]]
    C = [push[-1,0], push[a_pos,1]]
    return A, B, C, push

def rotate_point(xpoint, ypoint, angle, xcenter = 0, ycenter = 0, integer = False):
    """
    Introduction
    ------------
    This function will rotate a point in respect of another point (optionally
    given) by a certain angle.
    The outputs are the coordinates of the new point rotated
    
    
    Information
    -----------
        Author: Gaetano Camarda, Ph.D. Student in Structural Engineer
        Affiliation: University of Palermo
        e-mail: gaetano.camarda@unipa.it
        
                                                                     V1.0 2020
    Package needed
    --------------
    - numpy
        
    Input:
    ------
        xpoint, x coordinate of point to rotate
        ypoint, y coordinate of point to rotate
        angle, angle of rotation (in degree, it will be converted in rad)
    
    Optional input:
    ---------------
        xcenter, x coordinate of point as reference for rotation
        ycenter, y coordinate of point as reference for rotation
        integer, if "True" return the output as integer value
    """
    from numpy import cos
    from numpy import sin
    from numpy import round_
    x1 = xpoint - xcenter
    y1 = ypoint - ycenter
    x2 = x1 * cos(angle) - y1 * sin(angle)
    y2 = x1 * sin(angle) + y1 * cos(angle)
    newx = round_(x2 + xcenter,decimals = 3)
    newy = round_(y2 + ycenter,decimals = 3)
    if integer == True:
        return int(newx), int(newy)
    else:
        return newx, newy 

def section_creator(sectag, matcore, matcover, matsteel, d, b, cc, fi, bartop, 
                      nsuby, nsubz, fibot=0, barbot=0, fiside = 0, barside = 0, angle = 0, print_section = False):
    """
    Introduction
    ------------
    This function generate a fiber section for openseespy framework.
    It is implemented for RC section, with the possibility to Plot the section,
    and to rotate the RcSection by a given angle.
    As output the function will generate the give section on openseespy model.
    
     Information
     -----------
        Author: Gaetano Camarda, Ph.D. Student in Structural Engineer
        Affiliation: University of Palermo
        e-mail: gaetano.camarda@unipa.it
        
                                                                     V1.0 2020
    Package needed
    --------------
    - Openseespy
    - Openseespy.postprocessing
    - matplotlib
    - numpy
    
    
    Input:
    ------
        sectag, the sectiontag in the model
        matcore, material for confined core concrete
        matcover, material for concrete cover
        matsteel, material for reinforcement
        d, section depth
        b, section base length
        cc, cover length
        fi, top bar diameter
        bartop, number of top bar; if barbot is not specidied, bartop will be used
        nsuby, fiber subdivision along y axis
        nsubz, fiber subdivision along z axis
        
    Optional input:
    ---------------
        fibot, diameter of bottom reinforcement if different from top
        barbot, number of bottom reinforcement if different from top
        fiside, diameter of side bar
        barside, number of side bar (How to use:
                                     if for example you need 4 bars, you have to
                                     input barside = 2, because the other 2 bars
                                     are inserted in bartop and bottom)
        angle, angle if you need to rotate section, for example angle = 90
        print_section, plot the section defined
    """
    if fibot == 0:
        fibot = fi
    if barbot == 0:
        barbot = bartop
        
    import openseespy.opensees as ops
    import openseespy.postprocessing.ops_vis as opsv
    import matplotlib.pyplot as plt
    from numpy import power
    from numpy import pi
    ymax = b / 2
    zmax = d / 2
    ymin = ymax - cc
    zmin = zmax - cc
    Abar = pi * power(fi/2,2)
    Abarbot = pi * power(fibot/2,2)
    Abarside = pi * power(fiside/2,2)
    lside = d - (2*cc)
    bside = lside / (barside + 2) + (lside / (barside + 2))/(barside+2) # TODO: VERIFICARE COME SI COMPORTANO LE BARRE LATERAL
    angle = (angle * pi) /180
    pc = {
                        'a' : rotate_point(-ymin, -zmin, angle),
                        'b' : rotate_point(ymin, -zmin, angle),
                        'c' : rotate_point(ymin, zmin, angle),
                        'd' : rotate_point(-ymin, zmin, angle),
                        'e' : rotate_point(-ymax, -zmax, angle),
                        'f' : rotate_point(ymax, -zmax, angle),
                        'g' : rotate_point(ymax, zmax, angle),
                        'h' : rotate_point(-ymax, zmax, angle),
                        'b1': rotate_point(ymax, -zmin, angle),
                        'a1': rotate_point(-ymax, -zmin, angle),
                        'd1': rotate_point(-ymax, zmin, angle),
                        'c1': rotate_point(ymax, zmin, angle)
                        }
    fib_sec = [['section', 'Fiber', sectag],
                      ['patch','quad',matcore,nsuby, nsubz, pc['a'][0], pc['a'][1], pc['b'][0], pc['b'][1], pc['c'][0], pc['c'][1], pc['d'][0], pc['d'][1]], # core 1
                      ['patch','quad',matcover,nsuby, int(nsubz/2), pc['e'][0], pc['e'][1], pc['f'][0], pc['f'][1], pc['b1'][0], pc['b1'][1], pc['a1'][0], pc['a1'][1]], #  cover 2
                      ['patch','quad',matcover,nsuby, int(nsubz/2), pc['d1'][0], pc['d1'][1], pc['c1'][0], pc['c1'][1], pc['g'][0], pc['g'][1], pc['h'][0], pc['h'][1]], # cover 3
                      ['patch','quad',matcover,int(nsubz/2), nsuby, pc['a1'][0], pc['a1'][1], pc['a'][0], pc['a'][1], pc['d'][0], pc['d'][1], pc['d1'][0], pc['d1'][1]], # cover 4
                      ['patch','quad',matcover,int(nsubz/2), nsuby, pc['b'][0], pc['b'][1], pc['b1'][0], pc['b1'][1], pc['c1'][0], pc['c1'][1], pc['c'][0], pc['c'][1]] # cover 5
                      ]
    
    ops.section('Fiber', sectag)
    ops.patch('quad',matcore,nsuby, nsubz, pc['a'][0], pc['a'][1], pc['b'][0], pc['b'][1], pc['c'][0], pc['c'][1], pc['d'][0], pc['d'][1])
    ops.patch('quad',matcover,nsuby, int(nsubz/2), pc['e'][0], pc['e'][1], pc['f'][0], pc['f'][1], pc['b1'][0], pc['b1'][1], pc['a1'][0], pc['a1'][1])
    ops.patch('quad',matcover,nsuby, int(nsubz/2), pc['d1'][0], pc['d1'][1], pc['c1'][0], pc['c1'][1], pc['g'][0], pc['g'][1], pc['h'][0], pc['h'][1])
    ops.patch('quad',matcover,int(nsubz/2), nsuby, pc['a1'][0], pc['a1'][1], pc['a'][0], pc['a'][1], pc['d'][0], pc['d'][1], pc['d1'][0], pc['d1'][1])
    ops.patch('quad',matcover,int(nsubz/2), nsuby, pc['b'][0], pc['b'][1], pc['b1'][0], pc['b1'][1], pc['c1'][0], pc['c1'][1], pc['c'][0], pc['c'][1])
    
    if barbot == bartop:
        fib_sec.append(['layer', 'straight', matsteel, bartop, Abar, pc['d'][0], pc['d'][1], pc['c'][0], pc['c'][1]])
        fib_sec.append(['layer', 'straight', matsteel, barbot, Abarbot, pc['a'][0], pc['a'][1], pc['b'][0], pc['b'][1]])
        ops.layer('straight', matsteel, bartop, Abar, pc['d'][0], pc['d'][1], pc['c'][0], pc['c'][1]),
        ops.layer('straight', matsteel, barbot, Abarbot, pc['a'][0], pc['a'][1], pc['b'][0], pc['b'][1])
    elif bartop != barbot:
        fib_sec.append(['layer', 'straight', matsteel, bartop, Abar, pc['d'][0], pc['d'][1], pc['c'][0], pc['c'][1]])
        fib_sec.append( ['layer', 'straight', matsteel, barbot, Abarbot, pc['a'][0], pc['a'][1], pc['b'][0], pc['b'][1]])
        ops.layer('straight', matsteel, bartop, Abar, pc['d'][0], pc['d'][1], pc['c'][0], pc['c'][1]),
        ops.layer('straight', matsteel, barbot, Abarbot, pc['a'][0], pc['a'][1], pc['b'][0], pc['b'][1])
    if barside != 0:
        p1 = rotate_point(-ymin, -zmin + bside, angle)
        p2 = rotate_point(-ymin, zmin - bside, angle)
        fib_sec.append(['layer', 'straight', matsteel, barside, Abarside, p1[0], p1[1], p2[0], p2[1]])
        p3 = rotate_point(ymin, -zmin + bside, angle)
        p4 = rotate_point(ymin, zmin - bside, angle)
        fib_sec.append(['layer', 'straight', matsteel, barside, Abarside, p3[0], p3[1], p4[0], p4[1]])
        ops.layer('straight', matsteel, barside, Abarside, p1[0], p1[1], p2[0], p2[1])
        ops.layer('straight', matsteel, barside, Abarside, p3[0], p3[1], p4[0], p4[1])
    
    if print_section == True:
        # plt.style.use('classic')
        matcolor = ['r', 'darkgrey', 'lightgrey', 'w', 'w', 'w']
        opsv.plot_fiber_section(fib_sec, matcolor=matcolor)
        plt.axis('equal')
        del fib_sec
        
#### DISMISSED FOR NOW #####
# def PlotFrame(nodematrix,elementmatrix,displacement = None,scale = 1):
#     import matplotlib.pyplot as plt
#     plt.style.use('classic')
#     plt.figure()
#     if displacement is None:
#         plt.figure()
#         for i in range(len(elementmatrix[0])):
#             nodoi = elementmatrix[0][i]
#             nodoj = elementmatrix[1][i]
#             xx = (nodematrix[nodoi][0],nodematrix[nodoj][0])
#             yy = (nodematrix[nodoi][1],nodematrix[nodoj][1])
#             plt.text(nodematrix[nodoi][0], nodematrix[nodoi][1], nodoi)
#             plt.text(nodematrix[nodoj][0], nodematrix[nodoj][1], nodoj)
#             plt.text((nodematrix[nodoi][0]+nodematrix[nodoj][0])/2, (nodematrix[nodoi][1]+nodematrix[nodoj][1])/2, i, bbox=dict(facecolor='green', alpha=0.2))
#             plt.plot(xx, yy,'-k');
#     elif displacement is not None:
#             nodematrix_update = []
#             plt.figure()
#             for i in range(len(elementmatrix[0])):
#                 nodoi = elementmatrix[0][i]
#                 nodoj = elementmatrix[1][i]
#                 xx = (nodematrix[nodoi][0],nodematrix[nodoj][0])
#                 yy = (nodematrix[nodoi][1],nodematrix[nodoj][1])
#                 plt.text(nodematrix[nodoi][0], nodematrix[nodoi][1], nodoi)
#                 plt.text(nodematrix[nodoj][0], nodematrix[nodoj][1], nodoj)
#                 plt.text((nodematrix[nodoi][0]+nodematrix[nodoj][0])/2, (nodematrix[nodoi][1]+nodematrix[nodoj][1])/2, i,bbox=dict(facecolor='green', alpha=0.2))
#                 plt.plot(xx, yy,'-k');
#             txt = ('Node Displacement (Scale_factor: ' + str(scale) + ')')
#             plt.title(txt)
#             for i in range(len(nodematrix)):
#                 nodematrix_update.append([nodematrix[i][0]+displacement[i][0]*scale,nodematrix[i][1]+displacement[i][1]*scale])
#             for i in range(len(elementmatrix[0])):
#                 nodoi = elementmatrix[0][i]
#                 nodoj = elementmatrix[1][i]
#                 xx = (nodematrix_update[nodoi][0],nodematrix_update[nodoj][0])
#                 yy = (nodematrix_update[nodoi][1],nodematrix_update[nodoj][1])
#                 plt.plot(xx, yy,'-r');
                
# def PlotFrame_move(nodematrix,elementmatrix,displacement = None,scale = 1):
#     import matplotlib.pyplot as plt
#     plt.style.use('classic')
#     if displacement is None:
#         plt.figure()
#         for i in range(len(elementmatrix[0])):
#             nodoi = elementmatrix[0][i]
#             nodoj = elementmatrix[1][i]
#             xx = (nodematrix[nodoi][0],nodematrix[nodoj][0])
#             yy = (nodematrix[nodoi][1],nodematrix[nodoj][1])
#             plt.text(nodematrix[nodoi][0], nodematrix[nodoi][1], nodoi)
#             plt.text(nodematrix[nodoj][0], nodematrix[nodoj][1], nodoj)
#             plt.text((nodematrix[nodoi][0]+nodematrix[nodoj][0])/2, (nodematrix[nodoi][1]+nodematrix[nodoj][1])/2, i, bbox=dict(facecolor='green', alpha=0.2))
#             plt.plot(xx, yy,'-k');
#     elif displacement is not None:
#             nodematrix_update = []
#             plt.figure()
#             for i in range(len(elementmatrix[0])):
#                 nodoi = elementmatrix[0][i]
#                 nodoj = elementmatrix[1][i]
#                 xx = (nodematrix[nodoi][0],nodematrix[nodoj][0])
#                 yy = (nodematrix[nodoi][1],nodematrix[nodoj][1])
#                 plt.text(nodematrix[nodoi][0], nodematrix[nodoi][1], nodoi)
#                 plt.text(nodematrix[nodoj][0], nodematrix[nodoj][1], nodoj)
#                 plt.text((nodematrix[nodoi][0]+nodematrix[nodoj][0])/2, (nodematrix[nodoi][1]+nodematrix[nodoj][1])/2, i,bbox=dict(facecolor='green', alpha=0.2))
#                 plt.plot(xx, yy,'-k');
#             txt = ('Node Displacement (Scale_factor: ' + str(scale) + ')')
#             plt.title(txt)
#             for i in range(len(nodematrix)):
                nodematrix_update.append([nodematrix[i][0]+displacement[i][0]*scale,nodematrix[i][1]+displacement[i][1]*scale])
            for i in range(len(elementmatrix[0])):
                nodoi = elementmatrix[0][i]
                nodoj = elementmatrix[1][i]
                xx = (nodematrix_update[nodoi][0],nodematrix_update[nodoj][0])
                yy = (nodematrix_update[nodoi][1],nodematrix_update[nodoj][1])
                plt.plot(xx, yy,'-r');