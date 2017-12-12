# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 11:36:35 2016

@author: mmenoret
"""

def importmatfile(path):

    import mne
    import numpy as np
    import scipy.io as sio

    matf='.mat'
    mat = sio.loadmat((path['path']+path['ft_file']+matf), squeeze_me=True, struct_as_record=False)
    data = mat[path['ft_file']]
    ch_name = [l.encode('ascii') for l in data.label]
    ch_all = [l.encode('ascii') for l in data.elec.label]
    ch_type = data.label
    ch_type[:] = 'eeg'
    sfreq = data.fsample
    sel=np.ones(len(data.elec.chanpos)).astype(int)

    montage=mne.channels.Montage(pos=data.elec.chanpos,
                                 ch_names=ch_all,
                                 kind='BrainGraph',selection=sel)
    # Initialize an info structure
    info = mne.create_info(
        ch_names=ch_name,
        ch_types=ch_type,
        sfreq=sfreq,
        montage=montage)
    ###############################################################################
    # It is necessary to supply an "events" array in order to create an Epochs
    # object. This is of `shape(n_events, 3)` where the first column is the sample
    # number (time) of the event, the second column indicates the value from which
    # the transition is made from (only used when the new value is bigger than the
    # old one), and the third column is the new event value.
    
    # Create an event matrix: 10 events with alternating event codes
    events=np.zeros((len(np.array(data.trialinfo)),3),dtype=int)
    events[:,0]=data.trialinfo[:,0]
    events[:,2]=data.trialinfo[:,path['event']]
    # More information about the event codes:
    event_id = path['event_id']
    ###############################################################################
    # Finally, we must specify the beginning of an epoch (the end will be inferred
    # from the sampling frequency and n_samples)
    tmin=data.time[0][0]
    dat = sio.loadmat((path['path']+path['data']+matf), squeeze_me=True, struct_as_record=False)[path['data']]
    ###############################################################################
    # Now we can create the :class:`mne.EpochsArray` object
    
    custom_epochs = mne.EpochsArray(dat, info, events, tmin, event_id)
#   layout=mne.channels.find_layout(custom_epochs.info)
#   mne.viz.plot_layout(layout) 
    filename = 'W:/DATA/Comportement/badchannel.txt' 
    fin=open(filename,'r')
    badchannel=fin.read().splitlines()
    badchannels=set(badchannel) & set(custom_epochs.ch_names)
    custom_epochs.drop_channels(badchannels)
    custom_epochs.apply_baseline((-0.3,-0.1))
    return custom_epochs

###############################################################################
###############################################################################

    
listsuj=['begu'
         , 'brse', 'coma', 'coso', 'ersa', 'gudi',
    'imju', 'lema', 'leoc', 'mode', 'nofe', 'romu', 'sema',
    'toam', 'ausa', 'jean', 'rude', 'vapr', 'vasa' 
    ]

for suj in listsuj:    

    tache='naming';
    fold='W:/DATA/';
    path={}
    path['path']= fold+suj+'/'+tache+'/'
    path['ft_file']='dataclean'
    path['data']='trial'
    path['event']=5 #1 scrambled image 4 civilisation ou 5 biol
    path['event_id'] = dict(biol=1, nonbiol=2, scrambled=0)  

    epochs = importmatfile(path)
    epochs = epochs.copy().resample(200, npad='auto')
    epochs=epochs['biol','nonbiol']
    epochs.save(path['path']+'biol-nonbiol-epo.fif')
