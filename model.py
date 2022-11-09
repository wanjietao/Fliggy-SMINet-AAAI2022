linear_parent_scope = "linear"
dnn_parent_scope = "dnn"


def model_block(features, label_dict, fc_generator, is_training, keep_prob, params):
    # parse params
    black_list = params['black_list'] if 'black_list' in params else ""
    num_heads = params['num_heads'] if 'num_heads' in params else 4
    linear_key_dim = params['linear_key_dim'] if 'linear_key_dim' in params else 96
    linear_value_dim = params['linear_value_dim'] if 'linear_value_dim' in params else 96
    output_dim = params['output_dim'] if 'output_dim' in params else 96
    hidden_dim = params['hidden_dim'] if 'hidden_dim' in params else 96
    num_layer = params['num_layer'] if 'num_layer' in params else 3

    ########################################################
    # dnn
    outputs_dict = fc_generator.get_output_dict(features, black_list)

    for key in outputs_dict:
        tf.logging.info(key)
        tf.logging.info(outputs_dict[key])

    dense_feats = []
    for key in outputs_dict:
        if "is_dense" in key:
            tf.logging.info(key)
            dense_feats.append((key, outputs_dict[key]))

    dense_feats = [feat for _, feat in sorted(dense_feats, key=lambda x: x[0])]

    wide_dense_feats = []
    for key in outputs_dict:
        if "is_wide" in key:
            tf.logging.info(key)
            wide_dense_feats.append((key, outputs_dict[key]))

    wide_dense_feats = [feat for _, feat in sorted(wide_dense_feats, key=lambda x: x[0])]

    deep_feats = []
    for key in outputs_dict:
        if "is_deep" in key:
            tf.logging.info(key)
            deep_feats.append((key, outputs_dict[key]))

    deep_feats = [feat for _, feat in sorted(deep_feats, key=lambda x: x[0])]






    #########----spatial temporal recycle interest----#######
    shared_cm_item_list = outputs_dict["shared_cm_item_list"]
    shared_cm_item_list = tf.concat([tf.expand_dims(id, axis=1) for id in shared_cm_item_list], axis=1)

    #########----user recycle interest----#######
    shared_um_item_list = outputs_dict["shared_um_item_list"]
    shared_um_item_list = tf.concat([tf.expand_dims(id, axis=1) for id in shared_um_item_list], axis=1)

    #########----user group interest----#######
    shared_ug_item_list = outputs_dict["shared_ug_item_list"]
    shared_ug_item_list = tf.concat([tf.expand_dims(id, axis=1) for id in shared_ug_item_list], axis=1)


    #########----neighbor session----#######
    shared_nei_auction_id = outputs_dict["shared_nei_auction_id"]
    shared_nei_auction_id = tf.concat([tf.expand_dims(id, axis=1) for id in shared_nei_auction_id], axis=1)
    nei_auction_time = outputs_dict["nei2_nei_auction_time"]
    nei_auction_time = tf.concat(nei_auction_time, axis=1)

    #########----short interest----#######
    shared_auction_seq_id = outputs_dict["shared_auction_seq_id"]
    shared_auction_seq_id = tf.concat([tf.expand_dims(id, axis=1) for id in shared_auction_seq_id], axis=1)
    bhvs_auction_seq_time = outputs_dict["bhvs6_auction_seq_time"]
    bhvs_auction_seq_time = tf.concat(bhvs_auction_seq_time, axis=1)

    #########----long interest----#######
    shared_lng_item_id = outputs_dict["shared_lng_item_id"]
    shared_lng_item_id = tf.concat([tf.expand_dims(id, axis=1) for id in shared_lng_item_id], axis=1)
    lng1_lng_item_time = outputs_dict["lng6_lng_item_time"]
    lng1_lng_item_time = tf.concat(lng1_lng_item_time, axis=1)


    #target-vec
    activation_fn = tf.nn.relu
    target_vec = outputs_dict['target_vec']

    target_vec = tf.reshape(target_vec, [-1, 1, 32])
    _, _, target_vec_dim = target_vec.get_shape().as_list()
    _, seq_length, bhv_type_dim = shared_hotel_seq_type.get_shape().as_list()


    with tf.variable_scope('shared_ug_item_list'):

        input_group = tf.concat([hotel_id_vec, lp_id_vec, tag_id_vec, seller_id_vec, poi_id_vec, auction_id_vec, city_id_vec, content_id_vec, cate_id_vec, query_id_vec], axis=1)
        input_group = layers.batch_norm(input_group, is_training=is_training, activation_fn=None,
                                        variables_collections=[dnn_parent_scope])

        input_group = layers.fully_connected(input_group, 256, activation_fn=None, scope='input_short_ffn1',
                                             variables_collections=[dnn_parent_scope])
        input_group = layers.batch_norm(input_group, is_training=is_training, activation_fn=None,
                                        variables_collections=[dnn_parent_scope])
        input_group = layers.fully_connected(input_group, target_vec_dim, activation_fn=None, scope='input_short_ffn2',
                                             variables_collections=[dnn_parent_scope])


    with tf.variable_scope('shorterm_session_logit'):

        input_short = tf.concat([hotel_id_vec, lp_id_vec, tag_id_vec, seller_id_vec, poi_id_vec, auction_id_vec, city_id_vec, content_id_vec, cate_id_vec, query_id_vec], axis=1)
        input_short = layers.batch_norm(input_short, is_training=is_training, activation_fn=None,
                                        variables_collections=[dnn_parent_scope])

        input_short = layers.fully_connected(input_short, 256, activation_fn=None, scope='input_short_ffn1',
                                             variables_collections=[dnn_parent_scope])
        input_short = layers.batch_norm(input_short, is_training=is_training, activation_fn=None,
                                        variables_collections=[dnn_parent_scope])
        input_short = layers.fully_connected(input_short, target_vec_dim, activation_fn=None, scope='input_short_ffn2',
                                             variables_collections=[dnn_parent_scope])


    with tf.variable_scope('longterm_session_logit'):

        input_long = tf.concat([lng_cnt_vec, lng_prov_vec, lng_city_vec, lng_tag_vec, lng_poi_vec, lng_content_vec, lng_item_vec, lng_lp_vec, near_city_vec, near_poi_vec], axis=1)
        input_long = layers.batch_norm(input_long, is_training=is_training, activation_fn=None,
                                       variables_collections=[dnn_parent_scope])

        input_long = layers.fully_connected(input_long, 256, activation_fn=None, scope='input_long_ffn1',
                                            variables_collections=[dnn_parent_scope])
        input_long = layers.batch_norm(input_long, is_training=is_training, activation_fn=None,
                                       variables_collections=[dnn_parent_scope])
        input_long = layers.fully_connected(input_long, target_vec_dim, activation_fn=None, scope='input_long_ffn2',
                                            variables_collections=[dnn_parent_scope])


    #um cycle attention
    um_current_query = tf.concat([cureent_last_item, cureent_last_city, cureent_last_cate], axis=1)
    with tf.variable_scope("um_attention"):
        um_recycle_features = tf.concat([shared_um_item_list, shared_um_city_list, shared_um_cate_list], axis=2)
        um_trans_block = SelfAttentionPooling(
            num_heads=num_heads,
            key_mask=um_cycle_mask,
            query_mask=um_cycle_mask,
            length=4,
            linear_key_dim=linear_key_dim,
            linear_value_dim=linear_value_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            keep_prob=keep_prob
        )
        um_trans_output = um_trans_block.build(um_recycle_features, reuse=False,
                                                       scope='um_trans')  # (batch_size, 30, output_dim)

        um_attention_pool = time_attention_pooling(um_trans_output, um_current_query, um_cycle_mask,
                                                 um_cycle_mask, False, 'um_attention_pooling')

    #cm cycle attention
    cm_current_query = tf.concat([cureent_last_item, cureent_last_city, cureent_last_cate], axis=1)
    with tf.variable_scope("cm_attention"):
        cm_recycle_features = tf.concat([shared_um_item_list, shared_cm_item_list, shared_cm_mixtheme_list], axis=2)
        cm_trans_block = SelfAttentionPooling(
            num_heads=num_heads,
            key_mask=cm_cycle_mask,
            query_mask=cm_cycle_mask,
            length=4,
            linear_key_dim=linear_key_dim,
            linear_value_dim=linear_value_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            keep_prob=keep_prob
        )
        cm_trans_output = cm_trans_block.build(cm_recycle_features, reuse=False,
                                               scope='cm_trans')  # (batch_size, 30, output_dim)

        cm_attention_pool = time_attention_pooling(cm_trans_output, cm_current_query, cm_cycle_mask,
                                                   cm_cycle_mask, False, 'cm_attention_pooling')



    with tf.variable_scope('session_logit'):

        input = tf.concat([input_short, input_long, input_group, um_attention_pool, cm_attention_pool, cureent_last_item, cureent_last_city, cureent_last_cate], axis=1)


        input = layers.batch_norm(input, is_training=is_training, activation_fn=None,
                                  variables_collections=[dnn_parent_scope])


        input = layers.fully_connected(input, 512, activation_fn=None, scope='session_ffn1',
                                       variables_collections=[dnn_parent_scope])


        input = layers.batch_norm(input, is_training=is_training, activation_fn=None,
                                  variables_collections=[dnn_parent_scope])


        input = layers.fully_connected(input, 256, activation_fn=None, scope='session_ffn2',
                                       variables_collections=[dnn_parent_scope])


        input = layers.batch_norm(input, is_training=is_training, activation_fn=None,
                                  variables_collections=[dnn_parent_scope])

        input = layers.fully_connected(input, target_vec_dim, activation_fn=None, scope='session_ffn3',
                                       variables_collections=[dnn_parent_scope])


        target_vec = tf.reshape(target_vec, [-1, target_vec_dim])

        session_vec = tf.multiply(input_short, 1.0, name='session_vec')


        logit = tf.reduce_sum(tf.multiply(session_vec, target_vec), axis=1)
        logit = tf.reshape(logit + 1e-12, [-1,1])



    logit_dict = {}
    logit_dict['ctr'] = logit

    label_click = label_dict['click']
    label_click = tf.cast(tf.equal(label_click, '1'), tf.float32)
    label_click = tf.reshape(label_click, [-1, 1])
    label_dict['click'] = label_click

    return logit_dict, label_dict
