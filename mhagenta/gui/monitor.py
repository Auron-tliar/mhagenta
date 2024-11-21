import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from pydantic.dataclasses import dataclass
import dataclasses
import regex as re
from typing import ClassVar, Optional, Iterable
import asyncio

from mhagenta.utils import StatusReport


@dataclass
class AgentInfo:
    DECLARED: ClassVar[str] = 'DECLARED'
    _status_values: ClassVar[dict[str, int]] = {
        StatusReport.FINISHED: 0,
        StatusReport.TIMEOUT: 10,
        StatusReport.RUNNING: 20,
        StatusReport.READY: 30,
        StatusReport.CREATED: 40,
        DECLARED: 50,
        StatusReport.ERROR: 60
    }

    iid: str
    agent_id: str
    modules: dict[str, str]
    agent_status: str
    log_level_index: int
    logs: Optional[list[tuple[int, str, list[str]]]] = dataclasses.field(default_factory=list)

    def add_log(self, log: dict[str, str | int | bool | list[str]]) -> None:
        if log['status']:
            self._process_status(log['values'][-1].removeprefix('[status_upd]::'))
        else:
            self.logs.append((log['level'], log['module_id'], log['values']))
            if log['values'][-1] == 'Stopped!':
                self.modules[log['module_id']] = StatusReport.FINISHED

    def _process_status(self, status_str: str) -> None:
        statuses = status_str.split(',')
        self.modules['root'] = StatusReport.RUNNING
        self.agent_status = StatusReport.FINISHED
        for status in statuses:
            status = status.split(':')
            self.modules[status[0]] = status[1]
            self.agent_status = self.merge_status(self.agent_status, status[1])

    @classmethod
    def merge_status(cls, status1: str, status2: str) -> str:
        if cls._status_values[status1] >= cls._status_values[status2]:
            return status1
        else:
            return status2


class Monitor:
    class Colors:
        NORMAL = 'green3'
        WARNING = 'yellow1'
        ERROR = 'orangered1'
        OFF = 'gray50'

    DECLARED = 'DECLARED'
    status_colors: dict[str, str] = {
        DECLARED: Colors.WARNING,
        StatusReport.CREATED: Colors.WARNING,
        StatusReport.READY: Colors.WARNING,
        StatusReport.RUNNING: Colors.NORMAL,
        StatusReport.FINISHED: Colors.OFF,
        StatusReport.TIMEOUT: Colors.OFF,
        StatusReport.ERROR: Colors.ERROR
    }
    log_levels: list[str] = [
        'ALL',
        'Level 5',
        'DEBUG',
        'INFO',
        'WARNING',
        'ERROR',
        'CRITICAL',
    ]
    default_pattern: ClassVar[str] = (r'\[(?P<agent_time>\d+\.?\d*)\|(?P<mod_time>\d+\.?\d*)\|(?P<exec_time>\d+\.?\d*|-)\]'
                       r'\[(?P<level>.+)\]::\[(?P<agent_id>[^\[\]]+)\]\[(?P<module_id>[^\[\]]+)\]'
                       r'(\[(?P<extra_tags>.+)\])?::(?P<message>.+)')
    _level_str_to_int: ClassVar[dict[str, int]] = {
        'ALL': 0,
        'Level 5': 5,
        'DEBUG': 10,
        'INFO': 20,
        'WARNING': 30,
        'ERROR': 40,
        'CRITICAL': 50
    }

    def __init__(self, auto_select_new: bool = False, update_freq: float = 0.01) -> None:
        self._root = tk.Tk()

        font = tkFont.nametofont('TkDefaultFont').actual()
        self._font = tkFont.Font(self._root, family=font['family'], size=font['size'])

        self._root.columnconfigure(0, weight=0, minsize=32)
        self._root.columnconfigure(1, weight=1, minsize=32)
        self._root.rowconfigure(0, weight=1, minsize=64)

        self._agents_frame = ttk.Frame(self._root)
        self._modules_frame = ttk.Frame(self._root)
        self._modules_frame.rowconfigure(0, weight=0, minsize=32)
        self._modules_frame.rowconfigure(1, weight=0)
        self._modules_frame.rowconfigure(2, weight=1, minsize=32)
        self._modules_frame.columnconfigure(0, weight=1)

        self._agents_list = ttk.Treeview(self._agents_frame, show='tree', selectmode='browse',)
        for status, color in self.status_colors.items():
            self._agents_list.tag_configure(status, background=color)

        self._level_frame = ttk.Frame(self._modules_frame)
        self._level_frame.columnconfigure(0, weight=0)
        self._level_frame.columnconfigure(1, weight=0, minsize=4)
        self._level_frame.columnconfigure(2, weight=1)
        self._level_frame.rowconfigure(0, weight=1)
        ttk.Label(self._level_frame, text='Log level').grid(row=0, column=0, sticky=tk.W)
        self._log_level_selector = ttk.Combobox(self._level_frame, values=self.log_levels, width=10, state='readonly')
        self._log_level_selector.grid(row=0, column=2, sticky='w')
        self._log_level_selector.current(0)
        self._log_level_selector.bind('<<ComboboxSelected>>', self._on_level_changed)

        self._modules_list = ttk.Treeview(self._modules_frame, columns=(['Module ID']), show='tree', selectmode='browse')
        self._modules_list.column(0, width=32)
        for status, color in self.status_colors.items():
            self._agents_list.tag_configure(status, background=color)
            self._modules_list.tag_configure(status, background=color)

        column_names = ('Agent ts', 'Module ts', 'Execution ts', 'Level', 'Module ID', 'Extra tags', 'Message')
        self._logs_table = ttk.Treeview(self._modules_frame,
                                        columns=column_names,
                                        show='headings',
                                        selectmode='none')
        # self._logs_table.column('#0', minwidth=0, width=0, stretch=False, anchor='w')
        self._column_widths = list()
        for column in column_names:
            self._column_widths.append(self._font.measure(column) + 10)
            self._logs_table.heading(column, text=column, anchor='w')
            self._logs_table.column(column, stretch=False, width=self._column_widths[-1], anchor='w')
        self._column_widths[-1] = 512
        self._logs_table.column('Message', stretch=True, width=self._column_widths[-1], minwidth=64, anchor='w')

        self._agents_list.pack(expand=True, fill=tk.BOTH)

        self._modules_list.grid(row=0, column=0, sticky='nwe', pady=4)
        self._level_frame.grid(row=1, column=0, sticky='nwe', pady=4)
        self._logs_table.grid(row=2, column=0, sticky='nswe', pady=4)

        self._agents_frame.grid(column=0, row=0, sticky='nsw', padx=4, pady=4)
        self._modules_frame.grid(column=1, row=0, sticky='nswe', padx=4)

        self._agents_list.bind('<<TreeviewSelect>>', self._on_select_agent)
        self._modules_list.bind('<1>', self._on_module_click)
        self._root.protocol('WM_DELETE_WINDOW', self.close)

        self._agents: dict[str, AgentInfo] = dict()
        self._selected_agent: str | None = None
        self._selected_module: str | None = None
        self._selected_level: int = 0
        self._auto_select_new = auto_select_new
        self._update_freq = update_freq

        self._stop: bool = False

    async def run(self) -> None:
        self._root.update()
        while not self._stop:
            self._root.update()
            await asyncio.sleep(self._update_freq)

    def close(self) -> None:
        self._stop = True
        self._root.destroy()

    def _on_module_click(self, event: tk.Event) -> str:
        selection = self._modules_list.selection()
        item = self._modules_list.identify_row(event.y)
        if item in selection:
            self._modules_list.selection_remove(item)
            self._selected_module = None
            self._redraw_log_table()
            return 'break'
        else:
            self._selected_module = self._modules_list.item(item, option='text')
            self._redraw_log_table()

    def add_agent(self, agent_id, module_ids: Iterable[str]):
        iid = self._agents_list.insert('', tk.END, text=agent_id)
        self._agents[agent_id] = AgentInfo(
            iid=iid,
            agent_id=agent_id,
            modules={module_id: self.DECLARED for module_id in module_ids},
            agent_status=self.DECLARED,
            log_level_index=0
        )
        self._agents_list.item(iid, tags=self._agents[agent_id].agent_status)

        if len(self._agents) == 1:
            self._selected_agent = iid
            self._agents_list.selection_set(iid)
            self._on_select_agent()
        elif self._auto_select_new:
            self._selected_agent = iid
            self._agents_list.selection_set(iid)
            self._on_select_agent()

    def add_log(self, log_entry: str, log_format: str | None = None) -> None:
        if log_format is None:
            log = self.parse_log_default(log_entry)
        else:
            raise ValueError(f'Unsupported log format: {log_format}!')

        if log is None or log['agent_id'] not in self._agents:
            return

        agent = self._agents[log['agent_id']]
        agent.add_log(log)

        if log['status'] or log['values'][-1] == 'Stopped!':
            self._agents_list.item(agent.iid, tags=agent.agent_status)
            if self._selected_agent == agent.iid:
                for module in self._modules_list.get_children():
                    module_id = self._modules_list.item(module, option='text')
                    self._modules_list.item(module, tags=agent.modules[module_id])
        if (not log['status'] and
                self._selected_agent == agent.iid and
                log['level'] >= self._selected_level and
                (self._selected_module is None or self._selected_module == log['module_id'])):
            self._logs_table.insert('', tk.END, values=log['values'])
            for i, width in enumerate(self._column_widths):
                cur_width = self._font.measure(log['values'][i]) + 10
                if cur_width > width:
                    self._logs_table.column(i, width=cur_width)
                    self._column_widths[i] = cur_width


    def _on_select_agent(self, event: tk.Event | None = None) -> str:
        self._selected_agent = self._agents_list.selection()[0]
        self._modules_list.delete(*self._modules_list.get_children())
        self._selected_module = None
        self._logs_table.delete(*self._logs_table.get_children())

        if self._selected_agent is None:
            return 'continue'

        agent = self._agents[self._agents_list.item(self._selected_agent, option='text')]
        self._agents_list.item(self._selected_agent, tags=agent.agent_status)
        for module_id, status in agent.modules.items():
            self._modules_list.insert('', tk.END, text=module_id, tags=status)

        self._redraw_log_table()
        return 'break'

    def _on_level_changed(self, event: tk.Event) -> str:
        if self._log_level_selector.current() == self._selected_level:
            return 'break'

        self._selected_level = self._log_level_selector.current()
        self._redraw_log_table()

        return 'break'

    def _redraw_log_table(self) -> None:
        if self._selected_agent is None:
            return
        self._logs_table.delete(*self._logs_table.get_children())

        agent_id = self._agents_list.item(self._selected_agent, option='text')
        for level, module_id, log in self._agents[agent_id].logs:

            if level >= self._level_str_to_int[self._log_level_selector.get()] and (self._selected_module is None or self._selected_module == module_id):
                self._logs_table.insert('', tk.END, values=log)

    @classmethod
    def parse_log_default(cls, log_entry: str) -> dict[str, str | int | list[str]] | None:
        match = re.match(cls.default_pattern, log_entry, timeout=2)
        if match is None:
            return None

        return {
            'agent_id': match['agent_id'],
            'level': cls._level_str_to_int[match['level']],
            'module_id': match['module_id'],
            'status': match['module_id'] == 'root' and match['message'].startswith('[status_upd]::'),
            'values': [
                match['agent_time'],
                match['mod_time'],
                match['exec_time'],
                match['level'],
                match['module_id'],
                match['extra_tags'].replace('][', '.') if match['extra_tags'] is not None else '',
                match['message']
            ]}

