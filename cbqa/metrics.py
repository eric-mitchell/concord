import argparse


def tau_single_subject(predicates, truth_values, links):

    # follow bb's lead by only counting forward constraints
    antecedents = [l["source"] for l in links if l["direction"] == "forward"]
    consequents = [l["target"] for l in links if l["direction"] == "forward"]
    ground_truth_values = [
        l["weight"] for l in links if l["direction"] == "forward"
    ]

    beliefs = dict(zip(predicates, truth_values))

    applicable = 0
    violated = 0

    for i in range(len(antecedents)):
        ant = antecedents[i]

        if ant in beliefs and beliefs[ant]:
            cons = consequents[i]
            _, ct = ground_truth_values[i].split("_")

            if cons in beliefs:
                # wsa: if I follow the BB paper to the letter here, it defines
                # an applicable constraint as any whose antecedent is believed.
                #
                # this means the simple example below, taken as a whole, has
                # an inconsistency of around 0.023, simply because of the
                # volume of constraints that involve the premise 'IsA,tree'
                # about which no beliefs are held.  This seemed wrong to me,
                # so before calling a constraint applicable, there has to be
                # *some* belief about the consequent here.

                applicable += 1

                if (ct == "no" and beliefs[cons]) or (
                    ct == "yes" and not beliefs[cons]
                ):
                    violated += 1

    return violated, applicable


def consistency(subjects, predicates, truth_values, links):
    applicable = 0
    violated = 0

    for subject in list(set(subjects)):
        preds = [p for (p, s) in zip(predicates, subjects) if s == subject]

        tvs = [
            t
            for (t, p, s) in zip(truth_values, predicates, subjects)
            if s == subject
        ]

        v, a = tau_single_subject(preds, tvs, links)

        applicable += a
        violated += v

    return 1 - (violated / applicable)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run tests of functions provided by metrics"
    )

    parser.add_argument(
        "-c",
        "--constraints",
        help="Constraint file",
        type=argparse.FileType("r"),
    )

    args = parser.parse_args()

    ## simple test -- should come out to 50%
    subjects = ["A", "A", "A", "B", "B"]

    predicates = [
        "IsA,tree",
        "HasProperty,alive",
        "IsA,tracheophyte",
        "IsA,tree",
        "HasProperty,alive",
    ]

    truth_values = [True, True, False, True, True]

    print(consistency(subjects, predicates, truth_values, args.constraints))
