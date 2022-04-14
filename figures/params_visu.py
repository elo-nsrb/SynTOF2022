
params_ctx = {
        "model_dir": "ae_ctx_no_prp_GOOD/",
        "brain_region": "BA9-ctx",
        "list_no_mouse":[1, 2, 4, 5, 8, 9, 10, 11, 12, 15],
        "list_no_hu" :[3, 7, 13, 14],
         "list_no_mo" :[3, 7, 13, 14],
         "order" : [6, 3, 7, 13, 14,1, 2, 4, 5, 8, 9, 10, 11, 12, 15],

         "mapping" : {6:"A1", 3:"Mu1", 7:"Mu2", 13:"Mu3",
                      14:"Mu4", 1:"HuMo1", 
                      2:"HuMo2", 4:"HuMo3", 5:"HuMo4",8:"HuMo5",
                      9:"HuMo6", 10: "HuMo7", 11:"HuMo8",
                      12:"HuMo9", 15:"HuMo10"},
         "dico_color":{"HuMo9":"#7680B6", "A1":"#B1B1B1",
                        "Mu1":"#D64D86", "Mu2":"#D69ABF",#EED4B8
                         "Mu3":"#ECB4ED", "Mu4":"#FDB0C4", 
                         "HuMo1":"#E1C847", 
                        "HuMo2":"#DBDC8F", "HuMo3":"#D6F0D6",
                        "HuMo4":"#A7D0BB","HuMo5":"#499186",#"#0F9168"
                        "HuMo6":"#9BCFD7", "HuMo7":"#C4EEFF",
                        "HuMo8":"#9DB7DA",#8A9E00",#"#4BA849",
                        "HuMo10":"#4F4991"},
         "dico_offset_x":{"HuMo9":-2, "A1":0, "Mu1":6, "Mu2":6,
                         "Mu3":0, "Mu4":-2, "HuMo2":-2, 
                         "HuMo8":0, "HuMo3":0,"HuMo4":-3,"HuMo5":-10,
                            "HuMo6":3, "HuMo7":0, "HuMo1":3,
                            "HuMo10":-3},
         "dico_offset_y":{"HuMo9":0, "A1":-2, "Mu1":4, "Mu2":0,
                          "Mu3":0, "Mu4":0, "HuMo2":4, 
                          "HuMo8":0, "HuMo3":3,"HuMo4":0,"HuMo5":0,
                          "HuMo6":0, "HuMo7":-2, "HuMo1":4,
                                                    "HuMo10":-2},
         "color_map_species":{"hu":"#84BC9C", 
                              "mouse":"#F46197",
                              "mk":"#007991"},
         "color_batch_effect": ["#ffd8b1", "#e6194B",
                                "#2EC785", "#ffe119",
                                "#4363d8", "#911eb4"],
         "color_map_ba":{"BA9-ctx":"#EBC611", 
                            "Hipp":"#681B9E",
                            "DLCau-str":"#1C81EB"},
         "clusters_mo":["HuMo%d"%k for k in range(1,11)],
         "clusters_mu":["HuMu1", "A1"]

         }

params_hipp = {
        "model_dir": "ae_hipp_no_prp_GOOD/",
        "brain_region": "Hipp",
        "list_no_mouse":[2, 5, 6, 7, 10, 11],
        "list_no_hu":[1, 4, 8, 9, 14], 
         "list_no_mo" :[1,2,3,4,5,6,7,8,9,10,11,12, 13, 14],
         "order" :[3, 12, 13, 2,5, 6, 7, 10, 11, 1, 4, 8, 9, 14], 

         "mapping" :{3:"H-HuMu1", 12:"H-HuMu2", 13:"H-HuMu3", 2:"H-Hu1",
                    5:"H-Hu2", 6:"H-Hu3", 7:"H-Hu4", 10:"H-Hu5",11:"H-Hu6",
                    1:"H-Mu1", 4: "H-Mu2", 8:"H-Mu3",
                               9:"H-Mu4", 14:"H-Mu5"},
         "clusters_mo":[],
         "clusters_mu":["H-HuMu%d"%k for k in range(1,4)],
         "dico_color":{"H-HuMu1":'#8D7569', "H-HuMu2":"#B37D46",
                        "H-HuMu3":"#D98423", "H-Mu1":"#E8EAD1",
                     "H-Mu2":"#CDDA9F", "H-Mu3":"#B9CBB6",
                     "H-Mu4":"#6A9C21", 
                    "H-Mu5":"#91BD68", "H-Mu6":"#B7DEAE",
                    "H-Hu1":"#F4C9FF","H-Hu2":"#D791B5",#"#0F9168"
                    "H-Hu3":"#91498B", "H-Hu4":"#9C2020",
                    "H-Hu5":"#BA9FD8","H-Hu6":"#785A9B"
                                                },
         "color_map_species":{"hu":"#84BC9C", 
                            "mouse":"#F46197", 
                            "mk":"#007991"},
         "color_batch_effect": ["#ffd8b1", "#e6194B",
                                "#2EC785", "#ffe119", 
                                "#4363d8", "#911eb4"],
         "color_map_ba":{"BA9-ctx":"#EBC611",
                        "Hipp":"#681B9E",
                        "DLCau-str":"#1C81EB"},
         "dico_offset_x":{"H-HuMu1":5, "H-HuMu2":0, "H-HuMu3":0,
                         "H-Mu1":0, "H-Mu2":0, "H-Mu3":-2, "H-Mu4":0, 
                        "H-Mu5":0, "H-Mu6":-0,"H-Hu1":10,"H-Hu2":0,
                        "H-Hu3":17, "H-Hu5":0, "H-Hu4":0,"H-Hu6":-9, 
                                                    },
         "dico_offset_y":{"H-HuMu1":0, "H-HuMu2":0, "H-HuMu3":0, "H-Mu1":0,
                         "H-Mu2":0, "H-Mu3":-2, "H-Mu4":0, 
                         "H-Mu5":0, "H-Mu6":0,"H-Hu1":0,"H-Hu2":0,#"#0F9168"
                         "H-Hu3":0,"H-Hu4":10, "H-Hu5":0, "H-Hu6":0, 
                                                    }

         }
params_str = {
        "model_dir": "ae_str_no_prp_GOOD/",
        "brain_region": "DLCau-str",
        "list_no_mouse":[1,2,3,4,5,6,7,8,9,10,11,12, 13, 14, 15],

        "list_no_hu":[], 
         "list_no_mo" :[],
         "order" :[1,2,3,4,5,6,7,8,9,10,11,12, 13, 14, 15],

         "mapping" :{1:"S-HuMo1", 2:"S-HuMo2", 3:"S-HuMo3",
                     4:"S-HuMo4", 5:"S-HuMo5", 6:"S-HuMo6", 
                    7:"S-HuMo7", 8:"S-HuMo8", 9:"S-HuMo9",
                    10:"S-HuMo10",11:"S-HuMo11", 12: "S-HuMo12",
                    13:"S-HuMo13",
                    14:"S-HuMo14", 15:"S-HuMo15"},
         "clusters_mu":[],
         "clusters_mo":["S-HuMo%d"%k for k in range(1,16)],
         "dico_color":{"S-HuMo1":"#FFD20D", "S-HuMo2":"#BA7687",
                        "S-HuMo3":"#4FDE0B", "S-HuMo4":"#FFC9FB",#EED4B8
                         "S-HuMo5":"#D602F7", "S-HuMo6":"#F72902",
                         "S-HuMo7":"#F99339", 
                        "S-HuMo8":"#9077BB","S-HuMo9":"#FFD2AB",
                        "S-HuMo10":"#CDC9FF","S-HuMo11":"#F089C0",
                        "S-HuMo12":"#00ABF5",#"#0F9168"
                        "S-HuMo13":"#00F5C8", "S-HuMo14":"#C9FFFB",
                        "S-HuMo15":"#86AEC4",#8A9E00",#"#4BA849",
                                                    },
         "color_map_species":{"hu":"#84BC9C", 
                            "mouse":"#F46197", 
                            "mk":"#007991"},
         "color_batch_effect": ["#ffd8b1", "#e6194B",
                                "#2EC785", "#ffe119", 
                                "#4363d8", "#911eb4"],
         "color_map_ba":{"BA9-ctx":"#EBC611",
                        "Hipp":"#681B9E",
                        "DLCau-str":"#1C81EB"},
         "dico_offset_x":{"S-HuMo1":4, "S-HuMo2":-9, "S-HuMo3":0, 
                          "S-HuMo4":5, "S-HuMo5":-9, "S-HuMo6":12, 
                          "S-HuMo7":-5, "S-HuMo8":0, "S-HuMo9":0,
                          "S-HuMo10":-5,"S-HuMo11":12,#"#0F9168"
                         "S-HuMo12":12, "S-HuMo13":0,
                         "S-HuMo14":-5,"S-HuMo15":0
                                                    },
         "dico_offset_y":{"S-HuMo1":5, "S-HuMo2":-5, "S-HuMo3":0,
                        "S-HuMo4":-5, "S-HuMo5":0, "S-HuMo6":-4, 
                        "S-HuMo7":0, "S-HuMo8":5, "S-HuMo9":0,
                        "S-HuMo10":0,"S-HuMo11":0,#"#0F9168"
                        "S-HuMo12":0, "S-HuMo13":-12, "S-HuMo14":0,
                        "S-HuMo15":8
                                                    }
         }
