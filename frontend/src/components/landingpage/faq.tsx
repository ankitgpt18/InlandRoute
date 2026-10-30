"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import FaqIcon from "../icons/faqicon";

interface FAQItemProps {
  question: string;
  answer: string;
  index: number;
}

function FAQItem({ question, answer }: FAQItemProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <motion.div
      className={cn(
        "group rounded-xl border border-neutral-800/60",
        "transition-all duration-200 ease-in-out",
        isOpen
          ? "bg-neutral-900/60 border-neutral-700/80"
          : "hover:bg-neutral-900/30",
      )}
    >
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="flex w-full items-center justify-between gap-4 px-5 py-4"
      >
        <h3
          className={cn(
            "text-left text-sm sm:text-base font-medium transition-colors duration-200",
            "text-zinc-300",
            isOpen && "text-white",
          )}
        >
          {question}
        </h3>
        <motion.div
          animate={{
            rotate: isOpen ? 180 : 0,
            scale: isOpen ? 1.1 : 1,
          }}
          transition={{
            duration: 0.3,
            ease: "easeInOut",
          }}
          className={cn(
            "shrink-0 rounded-full p-0.5",
            "transition-colors duration-200",
            isOpen ? "text-white" : "text-zinc-500",
          )}
        >
          <ChevronDown className="h-4 w-4" />
        </motion.div>
      </button>
      <AnimatePresence initial={false}>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{
              height: "auto",
              opacity: 1,
              transition: {
                height: {
                  duration: 0.4,
                  ease: [0.04, 0.62, 0.23, 0.98],
                },
                opacity: {
                  duration: 0.25,
                  delay: 0.1,
                },
              },
            }}
            exit={{
              height: 0,
              opacity: 0,
              transition: {
                height: {
                  duration: 0.3,
                  ease: "easeInOut",
                },
                opacity: {
                  duration: 0.25,
                },
              },
            }}
          >
            <div className="px-5 pb-5 pt-1">
              <motion.p
                initial={{ y: -8, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                exit={{ y: -8, opacity: 0 }}
                transition={{
                  duration: 0.3,
                  ease: "easeOut",
                }}
                className="text-xs sm:text-sm leading-relaxed text-neutral-400"
              >
                {answer}
              </motion.p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function FAQ() {
  const faqs: Omit<FAQItemProps, "index">[] = [
    {
      question: "What is InlandRoute?",
      answer:
        "InlandRoute is an enterprise geospatial platform for predicting the navigability of India's National Waterways (NW-1 Ganga and NW-2 Brahmaputra). It combines multitemporal Sentinel-2 satellite imagery with CWC hydrological gauge telemetry to forecast fairway depths and identify bottleneck shoals.",
    },
    {
      question: "How is Least Available Depth (LAD) calculated?",
      answer:
        "LAD represents the minimum water depth available across the designated fairway channel at any given cross-section. The platform infers high-resolution riverbed bathymetry from satellite multispectral reflectance and hydrodynamic discharge models to forecast safe draft windows.",
    },
    {
      question: "Which National Waterways are currently covered?",
      answer:
        "The system monitors NW-1 (Prayagraj to Haldia, 1,620 km) and NW-2 (Dhubri to Sadiya, 891 km), with initial telemetry support for NW-16 (Barak River) and associated Indo-Bangladesh Protocol routes.",
    },
    {
      question: "What satellite cadence and imagery resolution is used?",
      answer:
        "We ingest Sentinel-2 A and B multispectral imagery at 10-meter spatial resolution with a 5-day revisit cadence, calibrated against continuous hydrological stage records from CWC gauge stations.",
    },
    {
      question: "Can fleet operators export route clearance advisories?",
      answer:
        "Yes. Navigators and operations teams can export longitudinal depth profile charts, LAD advisory bulletins, and waypoint coordinates in GeoJSON, CSV, and PDF formats for offline bridge use.",
    },
    {
      question: "Does the system support custom vessel draft safety margins?",
      answer:
        "Yes. Operators can define vessel-specific draft limits, safety clearances, and beam dimensions to receive early warning alerts before passing shallow braided sections.",
    },
  ];

  return (
    <section className="w-full bg-gradient-to-b from-transparent via-white/[0.02] to-transparent py-24">
      <motion.div
        initial={{ opacity: 0, y: 20, filter: "blur(8px)" }}
        whileInView={{ opacity: 1, y: 0, filter: "blur(0px)" }}
        viewport={{ once: true }}
        transition={{ duration: 0.8 }}
        className="container mx-auto px-4"
      >
        <motion.div className="mx-auto mb-12 max-w-7xl text-center">
          <FaqIcon />
          <h2 className="mx-auto mb-2 mt-6 max-w-3xl text-balance bg-gradient-to-br from-neutral-100 via-neutral-100 via-50% to-neutral-100/30 bg-clip-text py-2 text-center text-3xl font-medium leading-[1.15] tracking-tight text-transparent md:text-5xl">
            Frequently Asked Questions
          </h2>
          <p className="mb-8 text-balance bg-gradient-to-br from-white/70 via-white/70 to-white/30 bg-clip-text text-center text-[0.85rem] text-transparent sm:text-[0.92rem] lg:text-[1rem]">
            Platform architecture, satellite remote sensing, and fairway navigability standards
          </p>
        </motion.div>

        <div className="mx-auto max-w-2xl space-y-3">
          {faqs.map((faq, index) => (
            <FAQItem key={index} {...faq} index={index} />
          ))}
        </div>
      </motion.div>
    </section>
  );
}

export default FAQ;
