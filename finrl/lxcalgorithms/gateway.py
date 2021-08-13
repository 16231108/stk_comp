class Gateway:
    def __init__(
            self,
            agents = [],
            number = 0,
            last_time_step_stock = None,

    ):
        self.agents = agents
        self.agents_number = number
        self.last_time_step_stock = last_time_step_stock