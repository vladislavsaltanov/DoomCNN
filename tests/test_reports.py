from doomagent.reports import report_table_md


def summary(mean):
    return dict(reward=dict(mean=mean, lo=mean - 1, hi=mean + 1),
                kills=dict(mean=mean, lo=0, hi=1), shots=dict(mean=20.0, lo=0, hi=1),
                length=dict(mean=300.0, lo=0, hi=1), kills_per_shot=0.9, idle_rate=0.0,
                pressed_per_ep=25.0)


def test_table_has_header_rows_and_dashes_for_teacher():
    rep = {'teacher@id': summary(24.0),
           'bc@id': dict(summary(22.0), vs_teacher=(-2.0, -3.0, -1.0), latency_ms=1.5)}
    lines = report_table_md(rep).strip().split('\n')
    assert lines[0].startswith('| model') and len(lines) == 4
    assert '| teacher@id | 24.00 [23.00, 25.00] | -' in lines[2]
    assert '-2.00 [-3.00, -1.00]' in lines[3] and '1.5' in lines[3]
