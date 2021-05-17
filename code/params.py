# choose test-cases, std-scaler, test-type based on EDA


def get_test_params():
    test_cases = [
        "pglib_opf_case24_ieee_rts.m",
        "pglib_opf_case39_epri.m",
        "pglib_opf_case118_ieee.m",
        "pglib_opf_case162_ieee_dtc.m",
        "pglib_opf_case200_activ.m",
        "pglib_opf_case300_ieee.m",  ######
        # "pglib_opf_case30_ieee.m",
        # "pglib_opf_case57_ieee.m",
        # "pglib_opf_case73_ieee_rts.m",
        # "pglib_opf_case89_pegase.m",
        # "pglib_opf_case179_goc.m",
        # "pglib_opf_case240_pserc.m",
        # "pglib_opf_case500_goc.m",
        # "pglib_opf_case588_sdet.m",
        # "pglib_opf_case793_goc.m",
        # "pglib_opf_case1354_pegase.m",
        # "pglib_opf_case1888_rte.m",
        # "pglib_opf_case1951_rte.m",
        # "pglib_opf_case2000_goc.m",
        # "pglib_opf_case2312_goc.m",
        # "pglib_opf_case2383wp_k.m",
        # "pglib_opf_case2736sp_k.m",
        # "pglib_opf_case2737sop_k.m",
        # "pglib_opf_case2742_goc.m",
        # "pglib_opf_case2746wop_k.m",
        # "pglib_opf_case2746wop_k.m",
        # "pglib_opf_case2848_rte.m",
    ]

    dataset_size = 30000
    std_scaler = 0.09
    test_type = "default"

    return test_cases, dataset_size, std_scaler, test_type


def get_h_params(model_name):
    epochs = 10
    batch_size = None
    learning_rates = {"dense_net": 1e-3, "conv_net": 1e-3, "gcn": 1e-3, "gat": 1e-3}
    threshold = 0.1
    print_step = 1000
    return epochs, batch_size, learning_rates[model_name], threshold, print_step
